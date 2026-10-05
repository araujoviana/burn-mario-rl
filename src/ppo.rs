//! Actor-critic network (IMPALA or Nature CNN trunk, optional entity branch) and the PPO update.

use crate::entities::ENT_LEN;
use crate::env::{ACTIONS, STACK};
use crate::grid::{GRID_C, GRID_H, GRID_W};
use crate::obs::{self, ObsKind};
use burn::module::AutodiffModule;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::pool::{MaxPool2d, MaxPool2dConfig};
use burn::nn::{Linear, LinearConfig, PaddingConfig2d};
use burn::optim::{GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::activation::{log_softmax, relu, softmax};
use burn::tensor::backend::AutodiffBackend;

pub const NUM_ACTIONS: usize = ACTIONS.len();

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TrunkKind {
    Nature,
    Impala,
    /// Small convnet over the RAM tile grid (`grid.rs`); needs `obs::ObsKind::Grid`.
    Grid,
    /// The same grid through two dense layers: much cheaper on a CPU backend than the convs.
    GridMlp,
}

impl TrunkKind {
    pub fn obs_kind(self) -> ObsKind {
        if matches!(self, TrunkKind::Grid | TrunkKind::GridMlp) { ObsKind::Grid } else { ObsKind::Pixels }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct NetConfig {
    pub trunk: TrunkKind,
    /// Channel multiplier for the IMPALA trunk (1 = 16/32/32).
    pub width: usize,
    pub entities: bool,
}

impl Default for NetConfig {
    fn default() -> Self {
        Self { trunk: TrunkKind::Impala, width: 1, entities: true }
    }
}

impl NetConfig {
    /// Env vars TRUNK (nature|impala), WIDTH, ENTITIES (0|1).
    pub fn from_env() -> Self {
        let d = Self::default();
        let trunk = match std::env::var("TRUNK").as_deref() {
            Ok("nature") => TrunkKind::Nature,
            Ok("impala") => TrunkKind::Impala,
            Ok("grid") => TrunkKind::Grid,
            Ok("gridmlp") => TrunkKind::GridMlp,
            _ => d.trunk,
        };
        let width = std::env::var("WIDTH").ok().and_then(|v| v.parse().ok()).unwrap_or(d.width);
        let entities = std::env::var("ENTITIES").ok().map(|v| v != "0").unwrap_or(d.entities);
        Self { trunk, width, entities }
    }

    pub fn to_line(&self) -> String {
        format!("{} {} {}", match self.trunk { TrunkKind::Nature => "nature", TrunkKind::Impala => "impala", TrunkKind::Grid => "grid", TrunkKind::GridMlp => "gridmlp" }, self.width, self.entities as u8)
    }

    pub fn from_line(line: &str) -> Option<Self> {
        let f: Vec<&str> = line.split_whitespace().collect();
        let trunk = match *f.first()? {
            "nature" => TrunkKind::Nature,
            "impala" => TrunkKind::Impala,
            "grid" => TrunkKind::Grid,
            "gridmlp" => TrunkKind::GridMlp,
            _ => return None,
        };
        Some(Self { trunk, width: f.get(1)?.parse().ok()?, entities: *f.get(2)? != "0" })
    }

    /// Reads `net.cfg` next to a checkpoint (written by train); falls back to `from_env`.
    pub fn for_checkpoint(path: &str) -> Self {
        let dir = std::path::Path::new(path).parent().unwrap_or(std::path::Path::new("."));
        std::fs::read_to_string(dir.join("net.cfg")).ok().and_then(|s| Self::from_line(&s)).unwrap_or_else(Self::from_env)
    }
}

#[derive(Module, Debug)]
pub struct NatureTrunk<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    conv3: Conv2d<B>,
    fc: Linear<B>,
}

impl<B: Backend> NatureTrunk<B> {
    fn new(device: &B::Device) -> Self {
        Self {
            conv1: Conv2dConfig::new([STACK, 32], [8, 8]).with_stride([4, 4]).init(device),
            conv2: Conv2dConfig::new([32, 64], [4, 4]).with_stride([2, 2]).init(device),
            conv3: Conv2dConfig::new([64, 64], [3, 3]).init(device),
            fc: LinearConfig::new(64 * 7 * 7, 512).init(device),
        }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        let x = relu(self.conv1.forward(x));
        let x = relu(self.conv2.forward(x));
        let x = relu(self.conv3.forward(x));
        let batch = x.dims()[0];
        relu(self.fc.forward(x.reshape([batch, 64 * 7 * 7])))
    }
}

#[derive(Module, Debug)]
pub struct ImpalaStage<B: Backend> {
    conv: Conv2d<B>,
    pool: MaxPool2d,
}

impl<B: Backend> ImpalaStage<B> {
    fn new(c_in: usize, c_out: usize, device: &B::Device) -> Self {
        Self {
            conv: Conv2dConfig::new([c_in, c_out], [3, 3]).with_padding(PaddingConfig2d::Same).init(device),
            pool: MaxPool2dConfig::new([3, 3]).with_strides([2, 2]).with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1)).init(),
        }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        self.pool.forward(relu(self.conv.forward(x)))
    }
}

/// IMPALA-CNN: three conv/pool stages. 84x84 input becomes 11x11 feature maps.
#[derive(Module, Debug)]
pub struct ImpalaTrunk<B: Backend> {
    stages: Vec<ImpalaStage<B>>,
    fc: Linear<B>,
}

impl<B: Backend> ImpalaTrunk<B> {
    fn new(width: usize, device: &B::Device) -> Self {
        let ch = [16 * width, 32 * width, 32 * width];
        let stages = vec![ImpalaStage::new(STACK, ch[0], device), ImpalaStage::new(ch[0], ch[1], device), ImpalaStage::new(ch[1], ch[2], device)];
        Self { stages, fc: LinearConfig::new(ch[2] * 11 * 11, 256).init(device) }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        let mut x = x;
        for s in &self.stages {
            x = s.forward(x);
        }
        let [batch, c, h, w] = x.dims();
        relu(self.fc.forward(relu(x).reshape([batch, c * h * w])))
    }
}

/// Two small convs over the `[GRID_C, GRID_H, GRID_W]` tile grid, then a 128-unit layer.
#[derive(Module, Debug)]
pub struct GridTrunk<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    fc: Linear<B>,
}

const GRID_OUT_H: usize = GRID_H.div_ceil(2);
const GRID_OUT_W: usize = GRID_W.div_ceil(2);

impl<B: Backend> GridTrunk<B> {
    fn new(width: usize, device: &B::Device) -> Self {
        let (c1, c2) = (16 * width, 32 * width);
        Self {
            conv1: Conv2dConfig::new([GRID_C, c1], [3, 3]).with_padding(PaddingConfig2d::Same).init(device),
            conv2: Conv2dConfig::new([c1, c2], [3, 3]).with_stride([2, 2]).with_padding(PaddingConfig2d::Same).init(device),
            fc: LinearConfig::new(c2 * GRID_OUT_H * GRID_OUT_W, 128).init(device),
        }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        let x = relu(self.conv1.forward(x));
        let x = relu(self.conv2.forward(x));
        let [batch, c, h, w] = x.dims();
        relu(self.fc.forward(x.reshape([batch, c * h * w])))
    }
}

#[derive(Module, Debug)]
pub struct GridMlpTrunk<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
}

impl<B: Backend> GridMlpTrunk<B> {
    fn new(width: usize, device: &B::Device) -> Self {
        Self { fc1: LinearConfig::new(GRID_C * GRID_H * GRID_W, 256 * width).init(device), fc2: LinearConfig::new(256 * width, 128).init(device) }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        let [batch, c, h, w] = x.dims();
        relu(self.fc2.forward(relu(self.fc1.forward(x.reshape([batch, c * h * w])))))
    }
}

#[derive(Module, Debug)]
pub enum Trunk<B: Backend> {
    Nature(NatureTrunk<B>),
    Impala(ImpalaTrunk<B>),
    Grid(GridTrunk<B>),
    GridMlp(GridMlpTrunk<B>),
}

const ENT_HIDDEN: usize = 64;

#[derive(Module, Debug)]
pub struct ActorCritic<B: Backend> {
    trunk: Trunk<B>,
    ent_fc: Option<Linear<B>>,
    actor: Linear<B>,
    critic: Linear<B>,
}

impl<B: Backend> ActorCritic<B> {
    pub fn new(cfg: &NetConfig, device: &B::Device) -> Self {
        let (trunk, feat) = match cfg.trunk {
            TrunkKind::Nature => (Trunk::Nature(NatureTrunk::new(device)), 512),
            TrunkKind::Impala => (Trunk::Impala(ImpalaTrunk::new(cfg.width, device)), 256),
            TrunkKind::Grid => (Trunk::Grid(GridTrunk::new(cfg.width, device)), 128),
            TrunkKind::GridMlp => (Trunk::GridMlp(GridMlpTrunk::new(cfg.width, device)), 128),
        };
        let (ent_fc, fused) = if cfg.entities {
            (Some(LinearConfig::new(ENT_LEN, ENT_HIDDEN).init(device)), feat + ENT_HIDDEN)
        } else {
            (None, feat)
        };
        Self { trunk, ent_fc, actor: LinearConfig::new(fused, NUM_ACTIONS).init(device), critic: LinearConfig::new(fused, 1).init(device) }
    }

    /// `obs` holds raw bytes `[batch, C, H, W]` (see `obs::dims`) scaled to 0..1 here; `ent` is `[batch, ENT_LEN]`.
    pub fn forward(&self, obs: Tensor<B, 4>, ent: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 1>) {
        let x = obs / 255.0;
        let mut feat = match &self.trunk {
            Trunk::Nature(t) => t.forward(x),
            Trunk::Impala(t) => t.forward(x),
            Trunk::Grid(t) => t.forward(x),
            Trunk::GridMlp(t) => t.forward(x),
        };
        if let Some(fc) = &self.ent_fc {
            feat = Tensor::cat(vec![feat, relu(fc.forward(ent))], 1);
        }
        (self.actor.forward(feat.clone()), self.critic.forward(feat).squeeze_dim::<1>(1))
    }
}

pub fn ent_tensor<B: Backend>(floats: &[f32], batch: usize, device: &B::Device) -> Tensor<B, 2> {
    debug_assert_eq!(floats.len(), batch * ENT_LEN);
    Tensor::from_data(TensorData::new(floats.to_vec(), [batch, ENT_LEN]), device)
}

pub fn obs_tensor<B: Backend>(bytes: &[u8], batch: usize, device: &B::Device) -> Tensor<B, 4> {
    let (c, h, w) = obs::dims();
    debug_assert_eq!(bytes.len(), batch * c * h * w);
    let floats: Vec<f32> = bytes.iter().map(|&b| b as f32).collect();
    Tensor::from_data(TensorData::new(floats, [batch, c, h, w]), device)
}

pub struct Hyper {
    pub lr: f64,
    pub clip: f32,
    pub value_coef: f32,
    pub entropy_coef: f32,
    pub epochs: usize,
    pub minibatches: usize,
    /// Weight of the augmentation consistency term (0 = off).
    pub aug_coef: f32,
}

/// One minibatch of rollout data.
pub struct Batch<'a> {
    pub obs: &'a [u8],
    pub ent: &'a [f32],
    pub actions: &'a [i64],
    pub old_logp: &'a [f32],
    pub advantages: &'a [f32],
    pub returns: &'a [f32],
    /// Randomly shifted copy of `obs`; enables the consistency term.
    pub aug_obs: Option<&'a [u8]>,
}

pub struct UpdateStats {
    pub policy_loss: f32,
    pub value_loss: f32,
    pub entropy: f32,
    pub aug_loss: f32,
    /// Mean of (ratio - 1 - log ratio): how far the update moved the policy.
    pub approx_kl: f32,
    /// Share of samples whose ratio left the clip range.
    pub clip_frac: f32,
    /// 1 - var(returns - value) / var(returns): 1 = the critic explains the returns.
    pub explained_var: f32,
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

    let ent = ent_tensor::<B>(batch.ent, n, device);
    let (logits, value) = model.forward(obs, ent.clone());
    let log_probs = log_softmax(logits.clone(), 1);
    let logp = log_probs.clone().gather(1, actions).squeeze_dim::<1>(1);
    let log_ratio = logp - old_logp;
    let ratio = log_ratio.clone().exp();
    let approx_kl: f32 = (ratio.clone().detach() - 1.0 - log_ratio.detach()).mean().into_scalar().elem();
    let clip_frac: f32 = ratio.clone().detach().sub_scalar(1.0).abs().greater_elem(hyper.clip).float().mean().into_scalar().elem();
    let explained_var = {
        let diff = returns.clone() - value.clone().detach();
        let var = |t: Tensor<B, 1>| -> f32 { (t.clone().powf_scalar(2.0).mean() - t.mean().powf_scalar(2.0)).into_scalar().elem() };
        let vr = var(returns.clone());
        if vr > 1e-8 { 1.0 - var(diff) / vr } else { 0.0 }
    };
    let surr1 = ratio.clone() * adv.clone();
    let surr2 = ratio.clamp(1.0 - hyper.clip, 1.0 + hyper.clip) * adv;
    let policy_loss = -surr1.min_pair(surr2).mean();
    let value_loss = (value.clone() - returns).powf_scalar(2.0).mean() * 0.5;
    let entropy = -(softmax(logits.clone(), 1) * log_probs).sum_dim(1).mean();
    // DrAC: the PPO loss sees clean frames; shifted frames only have to agree with them.
    let aug_loss = match batch.aug_obs {
        Some(aug) if hyper.aug_coef > 0.0 => {
            let (aug_logits, aug_value) = model.forward(obs_tensor::<B>(aug, n, device), ent);
            let clean_lp = log_softmax(logits.clone().detach(), 1);
            let kl = (clean_lp.clone().exp() * (clean_lp - log_softmax(aug_logits, 1))).sum_dim(1).mean();
            let value_gap = (aug_value - value.clone().detach()).powf_scalar(2.0).mean() * 0.5;
            Some(kl + value_gap)
        }
        _ => None,
    };
    let mut loss = policy_loss.clone() + value_loss.clone() * hyper.value_coef - entropy.clone() * hyper.entropy_coef;
    let aug_stat = aug_loss.as_ref().map(|a| a.clone().into_scalar().elem::<f32>()).unwrap_or(0.0);
    if let Some(a) = aug_loss {
        loss = loss + a * hyper.aug_coef;
    }

    let stats = UpdateStats {
        policy_loss: policy_loss.into_scalar().elem(),
        value_loss: value_loss.into_scalar().elem(),
        entropy: entropy.into_scalar().elem(),
        aug_loss: aug_stat,
        approx_kl,
        clip_frac,
        explained_var,
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

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    type T = NdArray;

    fn run(cfg: NetConfig) -> ([usize; 2], [usize; 1]) {
        let device = Default::default();
        let model = ActorCritic::<T>::new(&cfg, &device);
        let (c, h, w) = if cfg.trunk.obs_kind() == ObsKind::Grid { (GRID_C, GRID_H, GRID_W) } else { (STACK, crate::env::OBS_H, crate::env::OBS_W) };
        let obs = Tensor::<T, 4>::zeros([3, c, h, w], &device);
        let ent = Tensor::<T, 2>::zeros([3, ENT_LEN], &device);
        let (logits, value) = model.forward(obs, ent);
        (logits.dims(), value.dims())
    }

    #[test]
    fn impala_shapes() {
        let (l, v) = run(NetConfig { trunk: TrunkKind::Impala, width: 1, entities: true });
        assert_eq!(l, [3, NUM_ACTIONS]);
        assert_eq!(v, [3]);
    }

    #[test]
    fn impala_wide_without_entities() {
        let (l, v) = run(NetConfig { trunk: TrunkKind::Impala, width: 2, entities: false });
        assert_eq!(l, [3, NUM_ACTIONS]);
        assert_eq!(v, [3]);
    }

    #[test]
    fn nature_shapes() {
        let (l, v) = run(NetConfig { trunk: TrunkKind::Nature, width: 1, entities: true });
        assert_eq!(l, [3, NUM_ACTIONS]);
        assert_eq!(v, [3]);
    }

    #[test]
    fn grid_shapes() {
        let (l, v) = run(NetConfig { trunk: TrunkKind::Grid, width: 1, entities: true });
        assert_eq!(l, [3, NUM_ACTIONS]);
        assert_eq!(v, [3]);
    }

    #[test]
    fn grid_mlp_shapes() {
        let (l, v) = run(NetConfig { trunk: TrunkKind::GridMlp, width: 1, entities: true });
        assert_eq!(l, [3, NUM_ACTIONS]);
        assert_eq!(v, [3]);
    }

    #[test]
    fn config_line_round_trip() {
        let cfg = NetConfig { trunk: TrunkKind::Impala, width: 2, entities: false };
        let back = NetConfig::from_line(&cfg.to_line()).unwrap();
        assert_eq!((back.trunk, back.width, back.entities), (TrunkKind::Impala, 2, false));
    }

    #[test]
    fn ppo_step_runs_with_augmentation() {
        use burn::backend::Autodiff;
        use burn::optim::AdamConfig;
        type A = Autodiff<NdArray>;
        let device = Default::default();
        let cfg = NetConfig { trunk: TrunkKind::Nature, width: 1, entities: true };
        let model = ActorCritic::<A>::new(&cfg, &device);
        let mut optim = AdamConfig::new().init();
        let n = 4;
        let obs = vec![7u8; n * crate::env::OBS_LEN];
        let ent = vec![0.0f32; n * ENT_LEN];
        let batch = Batch {
            obs: &obs,
            ent: &ent,
            actions: &[0, 1, 2, 3],
            old_logp: &[-1.9; 4],
            advantages: &[0.1, -0.1, 0.2, -0.2],
            returns: &[0.0; 4],
            aug_obs: Some(&obs),
        };
        let hyper = Hyper { lr: 1e-4, clip: 0.2, value_coef: 0.5, entropy_coef: 0.01, epochs: 1, minibatches: 1, aug_coef: 0.1 };
        let (_, stats) = ppo_step(model, &mut optim, &hyper, &batch, &device);
        assert!(stats.policy_loss.is_finite() && stats.value_loss.is_finite());
        assert!(stats.aug_loss.abs() < 1e-5, "identical clean and augmented input gives zero consistency loss");
    }
}
