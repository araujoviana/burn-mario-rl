//! Runs a checkpoint until an episode dies with max_x in [LO, HI], then dumps the last frames
//! (one per agent step) as /tmp/death_<n>.ppm. Env: CKPT, LO, HI, FRAMES.
use burn::module::Module;
use burn::record::CompactRecorder;
use burn_mario_rl::backend::Inner;
use burn_mario_rl::env::{MarioEnv, Outcome};
use burn_mario_rl::ppo::{ActorCritic, obs_tensor, sample_action};
use std::collections::VecDeque;
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
    let (lo, hi, keep): (u16, u16, usize) = (cfg("LO", 300), cfg("HI", 500), cfg("FRAMES", 12));
    let mut rng = 99u64;
    for ep in 0..200 {
        env.reset()?;
        let mut recent: VecDeque<Vec<u8>> = VecDeque::new();
        let (w, h) = (256, 224);
        loop {
            let (logits, _) = model.forward(obs_tensor::<Inner>(env.observation(), 1, &device));
            let logits = logits.into_data().to_vec::<f32>().map_err(|e| format!("{e:?}"))?;
            let action = sample_action(&logits, &mut rng).0;
            let r = env.step(action);
            env.emulator().with_frame(|rgb, _, _| {
                recent.push_back(rgb.to_vec());
                if recent.len() > keep { recent.pop_front(); }
            });
            if r.outcome != Outcome::Running {
                if r.outcome == Outcome::Died && (lo..=hi).contains(&env.max_x()) {
                    for (i, f) in recent.iter().enumerate() {
                        let mut out = format!("P6\n{w} {h}\n255\n").into_bytes();
                        out.extend_from_slice(f);
                        std::fs::write(format!("/tmp/death_{i:02}.ppm"), out).unwrap();
                    }
                    println!("episode {ep}: died at max_x {}, dumped {} frames", env.max_x(), recent.len());
                    return Ok(());
                }
                break;
            }
        }
    }
    Err("no matching death found".into())
}
