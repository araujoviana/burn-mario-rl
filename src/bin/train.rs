//! PPO training. Configure with env vars: ENVS, ROLLOUT, TOTAL_STEPS, LR, CORE, ROM, STATE, SEED.

use burn::module::{AutodiffModule, Module};
use burn::optim::AdamConfig;
use burn::optim::grad_clipping::GradientClippingConfig;
use burn::record::CompactRecorder;
use burn::tensor::Tensor;
use burn_mario_rl::backend::{Inner, Train};
use burn_mario_rl::env::{OBS_LEN, Outcome};
use burn_mario_rl::ppo::{ActorCritic, Batch, Hyper, NUM_ACTIONS, gae, obs_tensor, ppo_step, sample_action};
use burn_mario_rl::vec_env::{EpisodeInfo, VecEnv};
use std::collections::VecDeque;
use std::path::PathBuf;
use std::str::FromStr;
use std::time::Instant;

type B = Train;

fn cfg<T: FromStr>(name: &str, default: T) -> T {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn to_vec<const D: usize>(t: Tensor<Inner, D>) -> Vec<f32> {
    t.into_data().to_vec::<f32>().expect("f32 tensor")
}

fn main() -> Result<(), String> {
    let n: usize = cfg("ENVS", 10);
    let rollout: usize = cfg("ROLLOUT", 128);
    let total_steps: u64 = cfg("TOTAL_STEPS", 2_000_000);
    let core = PathBuf::from(cfg("CORE", "/usr/lib/libretro/snes9x_libretro.so".to_string()));
    let rom = PathBuf::from(cfg("ROM", "Super Mario World (USA).sfc".to_string()));
    let state = std::fs::read(cfg("STATE", "level1.state".to_string())).map_err(|e| format!("read state (run env_check first): {e}"))?;
    let mut rng: u64 = cfg("SEED", 0x2545F4914F6CDD1D);
    let hyper = Hyper { lr: cfg("LR", 2.5e-4), clip: 0.2, value_coef: 0.5, entropy_coef: cfg("ENTROPY", 0.01), epochs: 4, minibatches: 4 };
    let (gamma, lambda) = (0.99f32, 0.95f32);

    let device = Default::default();
    println!("backend {}", burn_mario_rl::backend::name());
    let mut model = ActorCritic::<B>::new(&device);
    let mut optim = AdamConfig::new().with_epsilon(1e-5).with_grad_clipping(Some(GradientClippingConfig::Norm(0.5))).init();
    std::fs::create_dir_all("checkpoints").map_err(|e| e.to_string())?;

    let mut envs = VecEnv::new(n, &core, &rom, &state)?;
    println!("{n} envs up, rollout {rollout}, {} steps per iteration", n * rollout);

    let batch_size = n * rollout;
    let mut obs_buf = vec![0u8; batch_size * OBS_LEN];
    let mut actions = vec![0i64; batch_size];
    let mut logps = vec![0f32; batch_size];
    let mut values = vec![0f32; batch_size];
    let mut rewards = vec![0f32; batch_size];
    let mut dones = vec![false; batch_size];

    let mut recent: VecDeque<EpisodeInfo> = VecDeque::new();
    let (mut steps_done, mut iter, mut best_mean, start) = (0u64, 0u32, f32::MIN, Instant::now());

    while steps_done < total_steps {
        let iter_start = Instant::now();
        let inference = model.valid();
        for t in 0..rollout {
            let range = t * n..(t + 1) * n;
            obs_buf[t * n * OBS_LEN..(t + 1) * n * OBS_LEN].copy_from_slice(&envs.obs);
            let (logits, value) = inference.forward(obs_tensor::<Inner>(&envs.obs, n, &device));
            let (logits, value) = (to_vec(logits), to_vec(value));
            let mut chosen = Vec::with_capacity(n);
            for e in 0..n {
                let (a, lp) = sample_action(&logits[e * NUM_ACTIONS..(e + 1) * NUM_ACTIONS], &mut rng);
                actions[t * n + e] = a as i64;
                logps[t * n + e] = lp;
                chosen.push(a);
            }
            values[range.clone()].copy_from_slice(&value);
            let step = envs.step(&chosen);
            rewards[range.clone()].copy_from_slice(&step.rewards);
            dones[range].copy_from_slice(&step.dones);
            for info in step.finished {
                recent.push_back(info);
                if recent.len() > 100 {
                    recent.pop_front();
                }
            }
        }
        let rollout_secs = iter_start.elapsed().as_secs_f32();
        let (_, last_value) = inference.forward(obs_tensor::<Inner>(&envs.obs, n, &device));
        let (adv, returns) = gae(&rewards, &values, &dones, &to_vec(last_value), n, gamma, lambda);

        // PPO epochs over shuffled minibatches.
        let mb = batch_size / hyper.minibatches;
        let mut order: Vec<usize> = (0..batch_size).collect();
        let mut last = None;
        for _ in 0..hyper.epochs {
            for i in (1..order.len()).rev() {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                order.swap(i, (rng % (i as u64 + 1)) as usize);
            }
            for chunk in order.chunks(mb) {
                let mut obs = Vec::with_capacity(chunk.len() * OBS_LEN);
                for &i in chunk {
                    obs.extend_from_slice(&obs_buf[i * OBS_LEN..(i + 1) * OBS_LEN]);
                }
                let pick = |src: &[f32]| chunk.iter().map(|&i| src[i]).collect::<Vec<f32>>();
                let mut adv_mb = pick(&adv);
                let mean = adv_mb.iter().sum::<f32>() / adv_mb.len() as f32;
                let std = (adv_mb.iter().map(|a| (a - mean).powi(2)).sum::<f32>() / adv_mb.len() as f32).sqrt() + 1e-8;
                adv_mb.iter_mut().for_each(|a| *a = (*a - mean) / std);
                let acts: Vec<i64> = chunk.iter().map(|&i| actions[i]).collect();
                let batch = Batch { obs: &obs, actions: &acts, old_logp: &pick(&logps), advantages: &adv_mb, returns: &pick(&returns) };
                let (m, stats) = ppo_step(model, &mut optim, &hyper, &batch, &device);
                model = m;
                last = Some(stats);
            }
        }

        steps_done += batch_size as u64;
        iter += 1;
        let stats = last.expect("at least one minibatch");
        let finished = recent.len().max(1) as f32;
        let mean_ret = recent.iter().map(|e| e.ret).sum::<f32>() / finished;
        let clear_rate = recent.iter().filter(|e| e.outcome == Outcome::Cleared).count() as f32 / finished;
        let mean_x = recent.iter().map(|e| e.max_x as f32).sum::<f32>() / finished;
        println!(
            "iter {iter:4} steps {steps_done:8} | rollout {rollout_secs:.1}s total {:.1}s | {:5.0} sps | return {mean_ret:7.1} clear {:3.0}% max_x {mean_x:6.0} | pl {:+.3} vl {:.3} ent {:.3}",
            iter_start.elapsed().as_secs_f32(),
            batch_size as f32 / iter_start.elapsed().as_secs_f32(),
            clear_rate * 100.0,
            stats.policy_loss,
            stats.value_loss,
            stats.entropy,
        );
        if recent.len() >= 20 && mean_ret > best_mean {
            best_mean = mean_ret;
            model.clone().save_file("checkpoints/best", &CompactRecorder::new()).map_err(|e| e.to_string())?;
        }
        if iter % 10 == 0 {
            model.clone().save_file("checkpoints/latest", &CompactRecorder::new()).map_err(|e| e.to_string())?;
        }
    }
    model.save_file("checkpoints/latest", &CompactRecorder::new()).map_err(|e| e.to_string())?;
    println!("done: {steps_done} steps in {:.0}s", start.elapsed().as_secs_f32());
    Ok(())
}
