//! Plays a checkpoint until one episode clears the level, saving an emulator state the first time
//! Mario passes each X threshold. Those states become extra start points for training.
//! Env: CKPT, OUT_DIR (default curriculum), THRESHOLDS (comma separated X positions), CORE.
use burn::module::Module;
use burn::record::CompactRecorder;
use burn_mario_rl::backend::Inner;
use burn_mario_rl::env::{MarioEnv, Outcome};
use burn_mario_rl::ppo::{ActorCritic, obs_tensor, sample_action};
use std::path::PathBuf;

fn cfg<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn main() -> Result<(), String> {
    let device = Default::default();
    let model = ActorCritic::<Inner>::new(&device)
        .load_file(cfg("CKPT", "checkpoints/best".to_string()), &CompactRecorder::new(), &device)
        .map_err(|e| e.to_string())?;
    let state = std::fs::read("level1.state").map_err(|e| e.to_string())?;
    let core = PathBuf::from(cfg("CORE", "/usr/lib/libretro/snes9x_libretro.so".to_string()));
    let mut env = MarioEnv::new(&core, &PathBuf::from("Super Mario World (USA).sfc"), state)?;
    let thresholds: Vec<u16> = cfg("THRESHOLDS", "100,200,300,500,900,1300,1450,1600,1900,2000,2100,2300,2700,3100,3500,3900,4300".to_string())
        .split(',')
        .filter_map(|t| t.trim().parse().ok())
        .collect();
    let out_dir: String = cfg("OUT_DIR", "curriculum".to_string());
    let mut rng = 4242u64;
    for ep in 0..100 {
        env.reset()?;
        let (mut next, mut saved): (usize, Vec<(u16, Vec<u8>)>) = (0, Vec::new());
        let outcome = loop {
            let (logits, _) = model.forward(obs_tensor::<Inner>(env.observation(), 1, &device));
            let logits = logits.into_data().to_vec::<f32>().map_err(|e| format!("{e:?}"))?;
            let r = env.step(sample_action(&logits, &mut rng).0);
            if r.outcome == Outcome::Running && next < thresholds.len() && env.x() >= thresholds[next] {
                saved.push((thresholds[next], env.emulator().save_state()?));
                next += 1;
            }
            if r.outcome != Outcome::Running {
                break r.outcome;
            }
        };
        println!("episode {ep}: {outcome:?}, reached {} of {} thresholds", saved.len(), thresholds.len());
        if outcome == Outcome::Cleared && saved.len() == thresholds.len() {
            std::fs::create_dir_all(&out_dir).map_err(|e| e.to_string())?;
            for (x, bytes) in &saved {
                std::fs::write(format!("{out_dir}/start_{x:05}.state"), bytes).map_err(|e| e.to_string())?;
            }
            println!("wrote {} states to {out_dir}", saved.len());
            return Ok(());
        }
    }
    Err("no cleared episode found".into())
}
