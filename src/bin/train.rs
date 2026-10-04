//! PPO training on a level set with PLR, entity inputs and DrAC augmentation.
//! Env vars: ENVS, ROLLOUT, TOTAL_STEPS, LR, CORE, ROM, SEED, INIT, CKPT_DIR, LEVELS, PLR, PLR_TEMP, PLR_RHO,
//! PLR_EMA, SPEED_K, AUG_PAD, AUG_COEF, EVAL_EVERY, EVAL_ENVS, EVAL_STEPS, TRUNK, WIDTH, ENTITIES.

use burn::module::{AutodiffModule, Module};
use burn::optim::AdamConfig;
use burn::optim::grad_clipping::GradientClippingConfig;
use burn::record::CompactRecorder;
use burn::tensor::Tensor;
use burn_mario_rl::augment::random_shift;
use burn_mario_rl::backend::{Inner, Train};
use burn_mario_rl::entities::ENT_LEN;
use burn_mario_rl::env::{OBS_LEN, Outcome};
use burn_mario_rl::levels::LevelSet;
use burn_mario_rl::plr::{LevelSampler, PlrConfig, level_scores};
use burn_mario_rl::ppo::{ActorCritic, Batch, Hyper, NUM_ACTIONS, NetConfig, ent_tensor, gae, obs_tensor, ppo_step, sample_action};
use burn_mario_rl::rng::next_u64;
use burn_mario_rl::vec_env::{EpisodeInfo, VecEnv};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::Arc;
use std::time::Instant;

type B = Train;

fn cfg<T: FromStr>(name: &str, default: T) -> T {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn to_vec<const D: usize>(t: Tensor<Inner, D>) -> Vec<f32> {
    t.into_data().to_vec::<f32>().expect("f32 tensor")
}

/// Per-level (clears, episodes, summed clear steps).
type EvalStats = BTreeMap<usize, (u32, u32, f32)>;

/// Run the held-out envs for `steps` agent steps with the current policy.
fn evaluate(model: &ActorCritic<Inner>, envs: &mut VecEnv, steps: usize, rng: &mut u64, device: &burn::tensor::Device<Inner>) -> EvalStats {
    let mut out = EvalStats::new();
    for _ in 0..steps {
        let (logits, _) = model.forward(obs_tensor::<Inner>(&envs.obs, envs.n, device), ent_tensor::<Inner>(&envs.ent, envs.n, device));
        let logits = to_vec(logits);
        let chosen: Vec<usize> = (0..envs.n).map(|e| sample_action(&logits[e * NUM_ACTIONS..(e + 1) * NUM_ACTIONS], rng).0).collect();
        for info in envs.step(&chosen).finished {
            let e = out.entry(info.level).or_insert((0, 0, 0.0));
            e.1 += 1;
            if info.outcome == Outcome::Cleared {
                e.0 += 1;
                e.2 += info.steps as f32;
            }
        }
    }
    out
}

fn clear_rate(eps: &VecDeque<EpisodeInfo>) -> f32 {
    eps.iter().filter(|e| e.outcome == Outcome::Cleared).count() as f32 / eps.len().max(1) as f32
}

fn main() -> Result<(), String> {
    let n: usize = cfg("ENVS", 10);
    let rollout: usize = cfg("ROLLOUT", 128);
    let total_steps: u64 = cfg("TOTAL_STEPS", 2_000_000);
    let core = PathBuf::from(cfg("CORE", "/usr/lib/libretro/snes9x_libretro.so".to_string()));
    let rom = PathBuf::from(cfg("ROM", "Super Mario World (USA).sfc".to_string()));

    let levels = Arc::new(LevelSet::load(Path::new(&cfg("LEVELS", "levels".to_string())))?);
    let train_ids = levels.train_ids();
    let held_ids = levels.held_out_ids();
    println!("levels: {} train, {} held out", train_ids.len(), held_ids.len());
    let plr_cfg = PlrConfig { enabled: cfg::<u8>("PLR", 1) == 1, temperature: cfg("PLR_TEMP", 0.3), rho: cfg("PLR_RHO", 0.1), ema: cfg("PLR_EMA", 0.3) };
    let sampler = LevelSampler::shared(train_ids.clone(), plr_cfg);
    let k_speed: f32 = cfg("SPEED_K", 1.0);
    let net_cfg = NetConfig::from_env();
    let aug_pad: usize = cfg("AUG_PAD", 4);
    let aug_coef: f32 = if aug_pad > 0 { cfg("AUG_COEF", 0.1) } else { 0.0 };
    println!("net {} | plr {} | speed k {k_speed} | aug pad {aug_pad} coef {aug_coef}", net_cfg.to_line(), plr_cfg.enabled);

    let seed: u64 = cfg("SEED", 0x2545F4914F6CDD1D);
    let mut rng = seed | 1;
    let hyper = Hyper { lr: cfg("LR", 2.5e-4), clip: 0.2, value_coef: 0.5, entropy_coef: cfg("ENTROPY", 0.01), epochs: cfg("EPOCHS", 4), minibatches: cfg("MINIBATCHES", 4), aug_coef };
    let (gamma, lambda): (f32, f32) = (cfg("GAMMA", 0.99), cfg("LAMBDA", 0.95));
    let ckpt_dir: String = cfg("CKPT_DIR", "checkpoints".to_string());
    // Keeps value targets near O(1); reported episode returns stay in raw reward units.
    let reward_scale: f32 = cfg("REWARD_SCALE", 0.1);

    let device = Default::default();
    println!("backend {}", burn_mario_rl::backend::name());
    let mut model = ActorCritic::<B>::new(&net_cfg, &device);
    let mut optim = AdamConfig::new().with_epsilon(1e-5).with_grad_clipping(Some(GradientClippingConfig::Norm(0.5))).init();
    if let Ok(path) = std::env::var("INIT") {
        model = model.load_file(path.clone(), &CompactRecorder::new(), &device).map_err(|e| format!("INIT {path}: {e}"))?;
        println!("warm start from {path}");
    }
    std::fs::create_dir_all(&ckpt_dir).map_err(|e| e.to_string())?;
    std::fs::write(format!("{ckpt_dir}/net.cfg"), net_cfg.to_line()).map_err(|e| e.to_string())?;

    let mut envs = VecEnv::new(n, &core, &rom, levels.clone(), sampler.clone(), k_speed, seed)?;
    println!("{n} envs up, rollout {rollout}, {} steps per iteration, {} updates per iteration, lr {}, gamma {gamma}, lambda {lambda}", n * rollout, hyper.epochs * hyper.minibatches, hyper.lr);

    let eval_every: u32 = cfg("EVAL_EVERY", 20);
    let eval_steps: usize = cfg("EVAL_STEPS", 600);
    let mut eval_envs = if eval_every > 0 && !held_ids.is_empty() {
        let eval_sampler = LevelSampler::shared(held_ids.clone(), PlrConfig { enabled: false, ..plr_cfg });
        Some(VecEnv::new(cfg("EVAL_ENVS", 4), &core, &rom, levels.clone(), eval_sampler, k_speed, seed ^ 0xABCD)?)
    } else {
        None
    };

    let batch_size = n * rollout;
    let mut obs_buf = vec![0u8; batch_size * OBS_LEN];
    let mut ent_buf = vec![0f32; batch_size * ENT_LEN];
    let mut level_buf = vec![0usize; batch_size];
    let mut actions = vec![0i64; batch_size];
    let mut logps = vec![0f32; batch_size];
    let mut values = vec![0f32; batch_size];
    let mut rewards = vec![0f32; batch_size];
    let mut dones = vec![false; batch_size];

    let mut recent: VecDeque<EpisodeInfo> = VecDeque::new();
    let mut per_level: HashMap<usize, VecDeque<EpisodeInfo>> = HashMap::new();
    let (mut steps_done, mut iter, mut best_rate, mut best_ret, start) = (0u64, 0u32, -1.0f32, f32::MIN, Instant::now());

    while steps_done < total_steps {
        let iter_start = Instant::now();
        let inference = model.valid();
        for t in 0..rollout {
            let range = t * n..(t + 1) * n;
            obs_buf[t * n * OBS_LEN..(t + 1) * n * OBS_LEN].copy_from_slice(&envs.obs);
            ent_buf[t * n * ENT_LEN..(t + 1) * n * ENT_LEN].copy_from_slice(&envs.ent);
            let (logits, value) = inference.forward(obs_tensor::<Inner>(&envs.obs, n, &device), ent_tensor::<Inner>(&envs.ent, n, &device));
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
            for (dst, r) in rewards[range.clone()].iter_mut().zip(&step.rewards) {
                *dst = r * reward_scale;
            }
            dones[range.clone()].copy_from_slice(&step.dones);
            level_buf[range].copy_from_slice(&step.step_levels);
            for info in step.finished {
                recent.push_back(info);
                if recent.len() > 100 {
                    recent.pop_front();
                }
                let q = per_level.entry(info.level).or_default();
                q.push_back(info);
                if q.len() > 50 {
                    q.pop_front();
                }
            }
        }
        let rollout_secs = iter_start.elapsed().as_secs_f32();
        let (_, last_value) = inference.forward(obs_tensor::<Inner>(&envs.obs, n, &device), ent_tensor::<Inner>(&envs.ent, n, &device));
        let (adv, returns) = gae(&rewards, &values, &dones, &to_vec(last_value), n, gamma, lambda);

        // PLR: levels where the value estimates were most wrong get sampled more.
        {
            let scores = level_scores(&adv, &level_buf, levels.levels.len());
            let mut s = sampler.lock().expect("sampler");
            for (l, score) in scores.iter().enumerate() {
                if let Some(score) = score {
                    s.update(l, *score);
                }
            }
        }

        // PPO epochs over shuffled minibatches.
        let mb = batch_size / hyper.minibatches;
        let mut order: Vec<usize> = (0..batch_size).collect();
        let mut last = None;
        for _ in 0..hyper.epochs {
            for i in (1..order.len()).rev() {
                order.swap(i, (next_u64(&mut rng) % (i as u64 + 1)) as usize);
            }
            for chunk in order.chunks(mb) {
                let mut obs = Vec::with_capacity(chunk.len() * OBS_LEN);
                let mut ent = Vec::with_capacity(chunk.len() * ENT_LEN);
                for &i in chunk {
                    obs.extend_from_slice(&obs_buf[i * OBS_LEN..(i + 1) * OBS_LEN]);
                    ent.extend_from_slice(&ent_buf[i * ENT_LEN..(i + 1) * ENT_LEN]);
                }
                let mut aug = vec![0u8; if aug_pad > 0 { obs.len() } else { 0 }];
                if aug_pad > 0 {
                    random_shift(&obs, &mut aug, chunk.len(), aug_pad, &mut rng);
                }
                let pick = |src: &[f32]| chunk.iter().map(|&i| src[i]).collect::<Vec<f32>>();
                let mut adv_mb = pick(&adv);
                let mean = adv_mb.iter().sum::<f32>() / adv_mb.len() as f32;
                let std = (adv_mb.iter().map(|a| (a - mean).powi(2)).sum::<f32>() / adv_mb.len() as f32).sqrt() + 1e-8;
                adv_mb.iter_mut().for_each(|a| *a = (*a - mean) / std);
                let acts: Vec<i64> = chunk.iter().map(|&i| actions[i]).collect();
                let batch = Batch {
                    obs: &obs,
                    ent: &ent,
                    actions: &acts,
                    old_logp: &pick(&logps),
                    advantages: &adv_mb,
                    returns: &pick(&returns),
                    aug_obs: (aug_pad > 0).then_some(&aug[..]),
                };
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
        let rate = clear_rate(&recent);
        let mean_x = recent.iter().map(|e| e.max_x as f32).sum::<f32>() / finished;
        let clears: Vec<&EpisodeInfo> = recent.iter().filter(|e| e.outcome == Outcome::Cleared).collect();
        let clear_time = clears.iter().map(|e| e.steps as f32).sum::<f32>() / clears.len().max(1) as f32;
        println!(
            "iter {iter:4} steps {steps_done:8} | rollout {rollout_secs:.1}s total {:.1}s | {:5.0} sps | return {mean_ret:6.1} clear {:3.0}% in {clear_time:4.0} steps (n={}) max_x {mean_x:5.0} | pl {:+.3} vl {:.3} ent {:.3} aug {:.4}",
            iter_start.elapsed().as_secs_f32(),
            batch_size as f32 / iter_start.elapsed().as_secs_f32(),
            rate * 100.0,
            recent.len(),
            stats.policy_loss,
            stats.value_loss,
            stats.entropy,
            stats.aug_loss,
        );
        if iter % 10 == 0 {
            let mut line = String::from("  per-level clear%:");
            for &l in &train_ids {
                if let Some(q) = per_level.get(&l) {
                    line += &format!(" L{}:{:.0}", levels.levels[l].id, clear_rate(q) * 100.0);
                }
            }
            println!("{line}");
            let weights = sampler.lock().expect("sampler").weights();
            let mut top: Vec<_> = weights.iter().map(|&(l, w)| (levels.levels[l].id, w)).collect();
            top.sort_by(|a, b| b.1.total_cmp(&a.1));
            let shown: Vec<String> = top.iter().take(5).map(|(id, w)| format!("L{id}:{:.0}%", w * 100.0)).collect();
            println!("  sampling weight (top 5): {}", shown.join(" "));
        }
        if let Some(eval) = eval_envs.as_mut().filter(|_| iter % eval_every == 0) {
            let result = evaluate(&model.valid(), eval, eval_steps, &mut rng, &device);
            let (mut clears, mut eps) = (0, 0);
            for (&l, &(c, e, t)) in &result {
                println!("  held-out level {} (tl {:#04x}): {c}/{e} cleared, mean clear time {:.0} steps", levels.levels[l].id, levels.levels[l].translevel, t / c.max(1) as f32);
                clears += c;
                eps += e;
            }
            println!("  held-out clear rate {clears}/{eps}");
        }
        if recent.len() >= 20 && (rate > best_rate || (rate == best_rate && mean_ret > best_ret)) {
            (best_rate, best_ret) = (rate, mean_ret);
            model.clone().save_file(format!("{ckpt_dir}/best"), &CompactRecorder::new()).map_err(|e| e.to_string())?;
        }
        if iter % 10 == 0 {
            model.clone().save_file(format!("{ckpt_dir}/latest"), &CompactRecorder::new()).map_err(|e| e.to_string())?;
        }
    }
    model.save_file(format!("{ckpt_dir}/latest"), &CompactRecorder::new()).map_err(|e| e.to_string())?;
    println!("done: {steps_done} steps in {:.0}s", start.elapsed().as_secs_f32());
    // Dropping two VecEnvs (train + eval) corrupts the heap while the cores unload; everything is saved,
    // so leave without running destructors.
    std::process::exit(0)
}
