//! Plays a checkpoint and records the first cleared episode to a video.
//! Env vars: CKPT (default checkpoints/best), OUT (default win.mp4), EPISODES, ARGMAX (1 = greedy),
//! ANY (1 = if nothing clears, keep the attempt that got furthest), CORE, ROM, LEVELS (dir), LEVEL (level id), SPEED_K.

use burn::module::Module;
use burn::record::CompactRecorder;
use burn_mario_rl::backend::Inner;
use burn_mario_rl::env::{MarioEnv, Outcome};
use burn_mario_rl::levels::LevelSet;
use burn_mario_rl::ppo::{ActorCritic, NUM_ACTIONS, NetConfig, ent_tensor, obs_tensor, sample_action};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::str::FromStr;

fn cfg<T: FromStr>(name: &str, default: T) -> T {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

struct Recorder {
    child: Child,
    stdin: ChildStdin,
}

impl Recorder {
    fn start(path: &str) -> Result<Self, String> {
        let mut child = Command::new("ffmpeg")
            .args(["-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "256x224", "-r", "60", "-i", "-"])
            .args(["-vf", "scale=768:672:flags=neighbor", "-c:v", "libx264", "-pix_fmt", "yuv420p", path])
            .stdin(Stdio::piped())
            .spawn()
            .map_err(|e| format!("ffmpeg: {e}"))?;
        let stdin = child.stdin.take().ok_or("ffmpeg stdin")?;
        Ok(Self { child, stdin })
    }

    fn finish(self) {
        drop(self.stdin);
        let mut child = self.child;
        let _ = child.wait();
    }
}

fn main() -> Result<(), String> {
    let core = PathBuf::from(cfg("CORE", "/usr/lib/libretro/snes9x_libretro.so".to_string()));
    let rom = PathBuf::from(cfg("ROM", "Super Mario World (USA).sfc".to_string()));
    let levels = Arc::new(LevelSet::load(Path::new(&cfg("LEVELS", "levels".to_string())))?);
    let level_id: usize = cfg("LEVEL", 0);
    let level = levels.levels.iter().position(|l| l.id == level_id).ok_or(format!("no level with id {level_id}"))?;
    let out: String = cfg("OUT", "win.mp4".to_string());
    let episodes: u32 = cfg("EPISODES", 20);
    let greedy: bool = cfg::<u8>("ARGMAX", 0) == 1;
    let keep_best_attempt: bool = cfg::<u8>("ANY", 0) == 1;
    let mut best_x = 0u16;
    let device = Default::default();
    let ckpt: String = cfg("CKPT", "checkpoints/best".to_string());
    let net_cfg = NetConfig::for_checkpoint(&ckpt);
    burn_mario_rl::obs::set(net_cfg.trunk.obs_kind());
    let model = ActorCritic::<Inner>::new(&net_cfg, &device)
        .load_file(ckpt, &CompactRecorder::new(), &device)
        .map_err(|e| format!("load checkpoint: {e}"))?;

    let mut env = MarioEnv::with_levels(&core, &rom, levels, cfg("SPEED_K", 1.0))?;
    let mut rng: u64 = cfg("SEED", 88172645463325252);
    let tmp = format!("{out}.tmp.mp4");
    let (mut cleared, mut saved, mut clear_steps) = (0u32, false, 0u32);
    for ep in 0..episodes {
        env.reset_to(level)?;
        let mut rec = if saved { None } else { Some(Recorder::start(&tmp)?) };
        let (mut ret, mut steps) = (0.0f32, 0u32);
        let outcome = loop {
            let (logits, _) = model.forward(obs_tensor::<Inner>(env.observation(), 1, &device), ent_tensor::<Inner>(env.entities(), 1, &device));
            let logits = logits.into_data().to_vec::<f32>().map_err(|e| format!("{e:?}"))?;
            let action = if greedy {
                (0..NUM_ACTIONS).max_by(|&a, &b| logits[a].total_cmp(&logits[b])).unwrap()
            } else {
                sample_action(&logits, &mut rng).0
            };
            let r = env.step_with(action, |emu| {
                if let Some(rec) = rec.as_mut() {
                    emu.with_frame(|rgb, _, _| {
                        let _ = rec.stdin.write_all(rgb);
                    });
                }
            });
            ret += r.reward;
            steps += 1;
            if r.outcome != Outcome::Running {
                break r.outcome;
            }
        };
        println!("episode {ep:3}: {outcome:?} in {steps} steps, return {ret:.1}, max_x {}", env.max_x());
        if let Some(rec) = rec {
            rec.finish();
            if outcome == Outcome::Cleared || (keep_best_attempt && env.max_x() > best_x) {
                best_x = best_x.max(env.max_x());
                std::fs::rename(&tmp, &out).map_err(|e| e.to_string())?;
                println!("  -> recorded {out} ({outcome:?})");
                saved = outcome == Outcome::Cleared;
            } else {
                let _ = std::fs::remove_file(&tmp);
            }
        }
        if outcome == Outcome::Cleared {
            cleared += 1;
            clear_steps += steps;
        }
    }
    println!("cleared {cleared}/{episodes} ({:.0}%){}", cleared as f32 * 100.0 / episodes as f32, if saved { "" } else { ", no video saved" });
    if cleared > 0 {
        println!("mean clear time {:.0} steps", clear_steps as f32 / cleared as f32);
    }
    Ok(())
}
