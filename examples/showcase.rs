//! Records the first cleared attempt of a checkpoint on one level for `scripts/showcase.py`: every emulator
//! frame (raw RGB), and per agent step the tile grid the network saw, its action probabilities and the critic's
//! value. Env: CKPT, LEVEL, LEVELS (dir), EPISODES, OUT (directory), SEED, CORE, FRAME_SKIP, ANY (1 = if nothing clears,
//! keep the attempt that got furthest).
use burn::module::Module;
use burn::record::CompactRecorder;
use burn_mario_rl::backend::Inner;
use burn_mario_rl::env::{ACTIONS, MarioEnv, Outcome};
use burn_mario_rl::grid::{CH_COIN, CH_FRIENDLY, CH_HOSTILE, CH_OBJECT, CH_SOLID, GRID_H, GRID_W};
use burn_mario_rl::levels::LevelSet;
use burn_mario_rl::ppo::{ActorCritic, NetConfig, ent_tensor, obs_tensor, sample_action};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

fn cfg<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// One digit per cell: 0 empty, 1 solid, 2 coin, 3 object, 4 hostile sprite, 5 friendly sprite.
fn grid_digits(obs: &[u8]) -> String {
    let plane = GRID_H * GRID_W;
    let mut s = String::with_capacity(plane);
    for i in 0..plane {
        let at = |ch: usize| obs[ch * plane + i] > 0;
        s.push(if at(CH_HOSTILE) { '4' } else if at(CH_FRIENDLY) { '5' } else if at(CH_SOLID) { '1' } else if at(CH_COIN) { '2' } else if at(CH_OBJECT) { '3' } else { '0' });
    }
    s
}

fn main() -> Result<(), String> {
    let device = Default::default();
    let ckpt: String = cfg("CKPT", "checkpoints/best".to_string());
    let net_cfg = NetConfig::for_checkpoint(&ckpt);
    burn_mario_rl::obs::set(net_cfg.trunk.obs_kind());
    let model = ActorCritic::<Inner>::new(&net_cfg, &device).load_file(ckpt, &CompactRecorder::new(), &device).map_err(|e| e.to_string())?;
    let core = PathBuf::from(cfg("CORE", "/usr/lib/libretro/snes9x_libretro.so".to_string()));
    let levels = Arc::new(LevelSet::load(Path::new(&cfg("LEVELS", "levels".to_string())))?);
    let level_id: usize = cfg("LEVEL", 7);
    let level = levels.levels.iter().position(|l| l.id == level_id).ok_or("no such level")?;
    let mut env = MarioEnv::with_levels(&core, &PathBuf::from("Super Mario World (USA).sfc"), levels, 1.0)?;
    let out_dir: String = cfg("OUT", "runs/showcase/L7".to_string());
    std::fs::create_dir_all(&out_dir).map_err(|e| e.to_string())?;
    let mut rng: u64 = cfg("SEED", 4242);
    let any = cfg::<u8>("ANY", 0) == 1;
    let mut best: Option<(u16, Vec<u8>, String, usize)> = None;
    for ep in 0..cfg("EPISODES", 30u32) {
        env.reset_to(level)?;
        let mut frames: Vec<u8> = Vec::new();
        let mut steps = String::new();
        let (mut nframes, mut t) = (0usize, 0usize);
        let outcome = loop {
            let (logits, value) = model.forward(obs_tensor::<Inner>(env.observation(), 1, &device), ent_tensor::<Inner>(env.entities(), 1, &device));
            let logits = logits.into_data().to_vec::<f32>().map_err(|e| format!("{e:?}"))?;
            let value = value.into_data().to_vec::<f32>().map_err(|e| format!("{e:?}"))?[0];
            let max = logits.iter().cloned().fold(f32::MIN, f32::max);
            let exps: Vec<f32> = logits.iter().map(|l| (l - max).exp()).collect();
            let sum: f32 = exps.iter().sum();
            let probs: Vec<String> = exps.iter().map(|e| format!("{:.3}", e / sum)).collect();
            let a = sample_action(&logits, &mut rng).0;
            let grid = grid_digits(env.observation());
            let x = env.x();
            let first = nframes;
            let r = env.step_with(a, |emu| {
                emu.with_frame(|rgb, _, _| frames.extend_from_slice(rgb));
                nframes += 1;
            });
            steps.push_str(&format!(r#"{{"t":{t},"f":{first},"n":{},"a":{a},"mask":{},"p":[{}],"v":{value:.2},"x":{x},"g":"{grid}"}}"#, nframes - first, ACTIONS[a], probs.join(",")));
            steps.push('\n');
            t += 1;
            if r.outcome != Outcome::Running {
                break r.outcome;
            }
        };
        println!("episode {ep}: {outcome:?} in {t} steps, max_x {}", env.max_x());
        if any && outcome != Outcome::Cleared && best.as_ref().is_none_or(|b| env.max_x() > b.0) {
            best = Some((env.max_x(), frames, steps, nframes));
        } else if outcome == Outcome::Cleared {
            std::fs::File::create(format!("{out_dir}/frames.rgb")).and_then(|mut f| f.write_all(&frames)).map_err(|e| e.to_string())?;
            std::fs::write(format!("{out_dir}/steps.jsonl"), steps).map_err(|e| e.to_string())?;
            println!("saved {out_dir} ({nframes} frames)");
            std::process::exit(0);
        }
    }
    if let Some((x, frames, steps, nframes)) = best {
        std::fs::File::create(format!("{out_dir}/frames.rgb")).and_then(|mut f| f.write_all(&frames)).map_err(|e| e.to_string())?;
        std::fs::write(format!("{out_dir}/steps.jsonl"), steps).map_err(|e| e.to_string())?;
        println!("no clear; saved the furthest attempt (x {x}, {nframes} frames) to {out_dir}");
        std::process::exit(0);
    }
    println!("no clear");
    std::process::exit(1);
}
