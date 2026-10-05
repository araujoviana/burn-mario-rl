//! Step-by-step trace of a checkpoint on one level: every action with its probabilities, the critic's
//! value, Mario's position and speed, and the reward. Writes JSONL to OUT and prints the tile grid the
//! network saw at the end of each episode (`.` empty, `#` solid, `o` coin, `+` object, `E` hostile,
//! `f` friendly, `M` Mario). Env: CKPT, LEVEL, EPISODES, OUT, SEED, ARGMAX, CORE.
use burn::module::Module;
use burn::record::CompactRecorder;
use burn_mario_rl::backend::Inner;
use burn_mario_rl::env::{MarioEnv, Outcome};
use burn_mario_rl::grid::{CH_COIN, CH_FRIENDLY, CH_HOSTILE, CH_OBJECT, CH_SOLID, GRID_H, GRID_W};
use burn_mario_rl::levels::LevelSet;
use burn_mario_rl::ppo::{ActorCritic, NUM_ACTIONS, NetConfig, ent_tensor, obs_tensor, sample_action};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

fn cfg<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn ascii_grid(obs: &[u8]) -> String {
    let plane = GRID_H * GRID_W;
    let mut out = String::new();
    for r in 0..GRID_H {
        for c in 0..GRID_W {
            let at = |ch: usize| obs[ch * plane + r * GRID_W + c] > 0;
            out.push(if r == 8 && c == 5 { 'M' } else if at(CH_HOSTILE) { 'E' } else if at(CH_FRIENDLY) { 'f' } else if at(CH_SOLID) { '#' } else if at(CH_COIN) { 'o' } else if at(CH_OBJECT) { '+' } else { '.' });
        }
        out.push('\n');
    }
    out
}

fn main() -> Result<(), String> {
    let device = Default::default();
    let ckpt: String = cfg("CKPT", "checkpoints/best".to_string());
    let net_cfg = NetConfig::for_checkpoint(&ckpt);
    burn_mario_rl::obs::set(net_cfg.trunk.obs_kind());
    let grid_mode = net_cfg.trunk.obs_kind() == burn_mario_rl::obs::ObsKind::Grid;
    let model = ActorCritic::<Inner>::new(&net_cfg, &device).load_file(ckpt, &CompactRecorder::new(), &device).map_err(|e| e.to_string())?;
    let core = PathBuf::from(cfg("CORE", "/usr/lib/libretro/snes9x_libretro.so".to_string()));
    let levels = Arc::new(LevelSet::load(Path::new(&cfg("LEVELS", "levels".to_string())))?);
    let level_id: usize = cfg("LEVEL", 7);
    let level = levels.levels.iter().position(|l| l.id == level_id).ok_or("no such level")?;
    let mut env = MarioEnv::with_levels(&core, &PathBuf::from("Super Mario World (USA).sfc"), levels, 1.0)?;
    let mut out = std::io::BufWriter::new(std::fs::File::create(cfg("OUT", "trace.jsonl".to_string())).map_err(|e| e.to_string())?);
    let greedy = cfg::<u8>("ARGMAX", 0) == 1;
    let mut rng: u64 = cfg("SEED", 777);
    for ep in 0..cfg("EPISODES", 3u32) {
        env.reset_to(level)?;
        let mut t = 0u32;
        let outcome = loop {
            let (logits, value) = model.forward(obs_tensor::<Inner>(env.observation(), 1, &device), ent_tensor::<Inner>(env.entities(), 1, &device));
            let logits = logits.into_data().to_vec::<f32>().map_err(|e| format!("{e:?}"))?;
            let value = value.into_data().to_vec::<f32>().map_err(|e| format!("{e:?}"))?[0];
            let max = logits.iter().cloned().fold(f32::MIN, f32::max);
            let exps: Vec<f32> = logits.iter().map(|l| (l - max).exp()).collect();
            let sum: f32 = exps.iter().sum();
            let probs: Vec<f32> = exps.iter().map(|e| e / sum).collect();
            let entropy: f32 = -probs.iter().map(|p| p * (p + 1e-9).ln()).sum::<f32>();
            let a = if greedy { (0..NUM_ACTIONS).max_by(|&i, &j| probs[i].total_cmp(&probs[j])).unwrap() } else { sample_action(&logits, &mut rng).0 };
            let r = env.step(a);
            let ram = env.emulator().ram();
            let (vx, vy) = (ram[burn_mario_rl::ram::PLAYER_SPEED_X] as i8, ram[burn_mario_rl::ram::PLAYER_SPEED_Y] as i8);
            let p: Vec<String> = probs.iter().map(|p| format!("{p:.3}")).collect();
            let _ = writeln!(
                out,
                r#"{{"ep":{ep},"t":{t},"a":{a},"p":[{}],"ent":{entropy:.3},"v":{value:.2},"x":{},"y":{},"vx":{vx},"vy":{vy},"r":{:.3}}}"#,
                p.join(","),
                env.x(),
                env.telemetry().end_y,
                r.reward
            );
            t += 1;
            if r.outcome != Outcome::Running {
                break r.outcome;
            }
        };
        let tele = env.telemetry();
        println!("episode {ep}: {outcome:?} after {t} steps, max_x {}, died by {} at ({}, {})", env.max_x(), tele.cause.name(), tele.end_x, tele.end_y);
        if grid_mode {
            println!("{}", ascii_grid(env.observation()));
        }
    }
    out.flush().map_err(|e| e.to_string())
}
