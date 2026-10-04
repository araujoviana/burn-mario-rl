//! Actor-critic network (Nature-DQN CNN trunk) and the PPO update.

use crate::env::{ACTIONS, OBS_H, OBS_LEN, OBS_W, STACK};
use burn::module::AutodiffModule;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{Linear, LinearConfig};
use burn::optim::{GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::activation::{log_softmax, relu, softmax};
use burn::tensor::backend::AutodiffBackend;

pub const NUM_ACTIONS: usize = ACTIONS.len();

#[derive(Module, Debug)]
pub struct ActorCritic<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    conv3: Conv2d<B>,
    fc: Linear<B>,
    actor: Linear<B>,
    critic: Linear<B>,
}

impl<B: Backend> ActorCritic<B> {
    pub fn new(device: &B::Device) -> Self {
        Self {
            conv1: Conv2dConfig::new([STACK, 32], [8, 8]).with_stride([4, 4]).init(device),
            conv2: Conv2dConfig::new([32, 64], [4, 4]).with_stride([2, 2]).init(device),
            conv3: Conv2dConfig::new([64, 64], [3, 3]).init(device),
            fc: LinearConfig::new(64 * 7 * 7, 512).init(device),
            actor: LinearConfig::new(512, NUM_ACTIONS).init(device),
            critic: LinearConfig::new(512, 1).init(device),
        }
    }

    /// `obs` holds raw bytes `[batch, STACK, OBS_H, OBS_W]` scaled to 0..1 here.
    pub fn forward(&self, obs: Tensor<B, 4>) -> (Tensor<B, 2>, Tensor<B, 1>) {
        let x = relu(self.conv1.forward(obs / 255.0));
        let x = relu(self.conv2.forward(x));
        let x = relu(self.conv3.forward(x));
        let batch = x.dims()[0];
        let x = relu(self.fc.forward(x.reshape([batch, 64 * 7 * 7])));
        (self.actor.forward(x.clone()), self.critic.forward(x).squeeze_dim::<1>(1))
    }
}

pub fn obs_tensor<B: Backend>(bytes: &[u8], batch: usize, device: &B::Device) -> Tensor<B, 4> {
    debug_assert_eq!(bytes.len(), batch * OBS_LEN);
    let floats: Vec<f32> = bytes.iter().map(|&b| b as f32).collect();
    Tensor::from_data(TensorData::new(floats, [batch, STACK, OBS_H, OBS_W]), device)
}

pub struct Hyper {
    pub lr: f64,
    pub clip: f32,
    pub value_coef: f32,
    pub entropy_coef: f32,
    pub epochs: usize,
    pub minibatches: usize,
}

/// One minibatch of rollout data.
pub struct Batch<'a> {
    pub obs: &'a [u8],
    pub actions: &'a [i64],
    pub old_logp: &'a [f32],
    pub advantages: &'a [f32],
    pub returns: &'a [f32],
}

pub struct UpdateStats {
    pub policy_loss: f32,
    pub value_loss: f32,
    pub entropy: f32,
}

pub fn ppo_step<B: AutodiffBackend, O: Optimizer<ActorCritic<B>, B>>(
    model: ActorCritic<B>,
    optim: &mut O,
    hyper: &Hyper,
    batch: &Batch,
    device: &B::Device,
) -> (ActorCritic<B>, UpdateStats) {
    let n = batch.actions.len();
    let obs = obs_tensor::<B>(batch.obs, n, device);
    let actions = Tensor::<B, 2, Int>::from_data(TensorData::new(batch.actions.to_vec(), [n, 1]), device);
    let old_logp = Tensor::<B, 1>::from_data(TensorData::new(batch.old_logp.to_vec(), [n]), device);
    let returns = Tensor::<B, 1>::from_data(TensorData::new(batch.returns.to_vec(), [n]), device);
    let adv = Tensor::<B, 1>::from_data(TensorData::new(batch.advantages.to_vec(), [n]), device);

    let (logits, value) = model.forward(obs);
    let log_probs = log_softmax(logits.clone(), 1);
    let logp = log_probs.clone().gather(1, actions).squeeze_dim::<1>(1);
    let ratio = (logp - old_logp).exp();
    let surr1 = ratio.clone() * adv.clone();
    let surr2 = ratio.clamp(1.0 - hyper.clip, 1.0 + hyper.clip) * adv;
    let policy_loss = -surr1.min_pair(surr2).mean();
    let value_loss = (value - returns).powf_scalar(2.0).mean() * 0.5;
    let entropy = -(softmax(logits, 1) * log_probs).sum_dim(1).mean();
    let loss = policy_loss.clone() + value_loss.clone() * hyper.value_coef - entropy.clone() * hyper.entropy_coef;

    let stats = UpdateStats {
        policy_loss: policy_loss.into_scalar().elem(),
        value_loss: value_loss.into_scalar().elem(),
        entropy: entropy.into_scalar().elem(),
    };
    let grads = GradientsParams::from_grads(loss.backward(), &model);
    (optim.step(hyper.lr, model, grads), stats)
}

/// Generalized advantage estimation over `[steps][envs]` flattened step-major.
pub fn gae(rewards: &[f32], values: &[f32], dones: &[bool], last_values: &[f32], envs: usize, gamma: f32, lambda: f32) -> (Vec<f32>, Vec<f32>) {
    let steps = rewards.len() / envs;
    let mut adv = vec![0.0f32; rewards.len()];
    let mut running = vec![0.0f32; envs];
    for t in (0..steps).rev() {
        for e in 0..envs {
            let i = t * envs + e;
            let next_value = if t + 1 == steps { last_values[e] } else { values[i + envs] };
            let not_done = if dones[i] { 0.0 } else { 1.0 };
            let delta = rewards[i] + gamma * next_value * not_done - values[i];
            running[e] = delta + gamma * lambda * not_done * running[e];
            adv[i] = running[e];
        }
    }
    let returns = adv.iter().zip(values).map(|(a, v)| a + v).collect();
    (adv, returns)
}

pub fn sample_action(logits: &[f32], rng: &mut u64) -> (usize, f32) {
    let max = logits.iter().cloned().fold(f32::MIN, f32::max);
    let exps: Vec<f32> = logits.iter().map(|l| (l - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    *rng ^= *rng << 13;
    *rng ^= *rng >> 7;
    *rng ^= *rng << 17;
    let mut u = (*rng >> 40) as f32 / (1u64 << 24) as f32 * sum;
    let mut action = exps.len() - 1;
    for (i, e) in exps.iter().enumerate() {
        if u < *e {
            action = i;
            break;
        }
        u -= e;
    }
    (action, (exps[action] / sum).ln())
}

pub fn inference_model<B: AutodiffBackend>(model: &ActorCritic<B>) -> ActorCritic<B::InnerBackend> {
    model.valid()
}
