//! Counts which actions a checkpoint picks on a level. Env: CKPT, LEVEL, EPISODES.
use burn::module::Module;
use burn::record::CompactRecorder;
use burn_mario_rl::backend::Inner;
use burn_mario_rl::env::{ACTIONS, MarioEnv, Outcome};
use burn_mario_rl::levels::LevelSet;
use burn_mario_rl::ppo::{ActorCritic, NUM_ACTIONS, NetConfig, ent_tensor, obs_tensor, sample_action};
use std::path::{Path, PathBuf};
use std::sync::Arc;

fn cfg<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn main() -> Result<(), String> {
    let device = Default::default();
    let ckpt: String = cfg("CKPT", "checkpoints/best".to_string());
    let net_cfg = NetConfig::for_checkpoint(&ckpt);
    burn_mario_rl::obs::set(net_cfg.trunk.obs_kind());
    let model = ActorCritic::<Inner>::new(&net_cfg, &device).load_file(ckpt, &CompactRecorder::new(), &device).map_err(|e| e.to_string())?;
    let core = PathBuf::from(cfg("CORE", "/usr/lib/libretro/snes9x_libretro.so".to_string()));
    let levels = Arc::new(LevelSet::load(Path::new("levels"))?);
    let level = levels.levels.iter().position(|l| l.id == cfg("LEVEL", 7usize)).ok_or("no such level")?;
    let mut env = MarioEnv::with_levels(&core, &PathBuf::from("Super Mario World (USA).sfc"), levels, 1.0)?;
    let (mut counts, mut rng) = ([0u32; NUM_ACTIONS], 12345u64);
    let mut run_steps = 0u32;
    for _ in 0..cfg("EPISODES", 10u32) {
        env.reset_to(level)?;
        loop {
            let (logits, _) = model.forward(obs_tensor::<Inner>(env.observation(), 1, &device), ent_tensor::<Inner>(env.entities(), 1, &device));
            let logits = logits.into_data().to_vec::<f32>().map_err(|e| format!("{e:?}"))?;
            let a = sample_action(&logits, &mut rng).0;
            counts[a] += 1;
            run_steps += (ACTIONS[a] & burn_mario_rl::emulator::button::Y != 0) as u32;
            if env.step(a).outcome != Outcome::Running { break; }
        }
    }
    let total: u32 = counts.iter().sum();
    let names = ["none", "right", "right+B", "right+Y", "right+Y+B", "B", "left", "down", "up", "right+A", "right+Y+A"];
    for (n, c) in names.iter().zip(counts).take(NUM_ACTIONS) { println!("{n:10} {:5.1}%", c as f32 * 100.0 / total as f32); }
    println!("run button held {:.1}% of steps", run_steps as f32 * 100.0 / total as f32);
    Ok(())
}
