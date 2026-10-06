//! Records many attempts of a checkpoint on one level as per-frame positions, plus the level's tile map,
//! for `scripts/ghosts.py` (the "all attempts at once" replay). Env: CKPT, LEVEL, EPISODES, OUT, CORE, FRAME_SKIP.
use burn::module::Module;
use burn::record::CompactRecorder;
use burn_mario_rl::backend::Inner;
use burn_mario_rl::env::{MarioEnv, Outcome};
use burn_mario_rl::levels::LevelSet;
use burn_mario_rl::ppo::{ActorCritic, NetConfig, ent_tensor, obs_tensor, sample_action};
use burn_mario_rl::ram;
use std::fmt::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

fn cfg<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// The level's whole Map16 layer as one string per row: `.` empty, `#` solid, `o` coin, `+` other object.
fn tile_rows(ram_bytes: &[u8]) -> Vec<String> {
    const SCREENS: usize = 19; // the buffer holds about 19 screens of 16x27 tiles
    (0..27)
        .map(|y| {
            (0..SCREENS * 16)
                .map(|x| {
                    let off = (x >> 4) * 0x1B0 + y * 16 + (x & 15);
                    if off >= 0x2000 {
                        return '#';
                    }
                    let id = (ram_bytes[0x1C800 + off] as usize) << 8 | ram_bytes[0xC800 + off] as usize;
                    match id {
                        0x25 => '.',
                        0x2B => 'o',
                        i if i >= 0x100 => '#',
                        _ => '+',
                    }
                })
                .collect()
        })
        .collect()
}

fn main() -> Result<(), String> {
    let device = Default::default();
    let ckpt: String = cfg("CKPT", "checkpoints/best".to_string());
    let net_cfg = NetConfig::for_checkpoint(&ckpt);
    burn_mario_rl::obs::set(net_cfg.trunk.obs_kind());
    let model = ActorCritic::<Inner>::new(&net_cfg, &device).load_file(ckpt, &CompactRecorder::new(), &device).map_err(|e| e.to_string())?;
    let core = PathBuf::from(cfg("CORE", "/usr/lib/libretro/snes9x_libretro.so".to_string()));
    let levels = Arc::new(LevelSet::load(Path::new("levels"))?);
    let id: usize = cfg("LEVEL", 3);
    let level = levels.levels.iter().position(|l| l.id == id).ok_or("no such level")?;
    let mut env = MarioEnv::with_levels(&core, &PathBuf::from("Super Mario World (USA).sfc"), levels, 1.0)?;
    env.reset_to(level)?;
    let rows = tile_rows(env.emulator().ram());
    let mut rng: u64 = cfg("SEED", 4242);
    let mut json = String::new();
    write!(json, r#"{{"level":{id},"frame_skip":{},"tiles":["#, burn_mario_rl::env::frame_skip()).unwrap();
    json.push_str(&rows.iter().map(|r| format!("\"{r}\"")).collect::<Vec<_>>().join(","));
    json.push_str(r#"],"episodes":["#);
    let n: u32 = cfg("EPISODES", 60);
    for ep in 0..n {
        env.reset_to(level)?;
        let (mut xs, mut ys): (Vec<u16>, Vec<u16>) = (Vec::new(), Vec::new());
        let outcome = loop {
            let (logits, _) = model.forward(obs_tensor::<Inner>(env.observation(), 1, &device), ent_tensor::<Inner>(env.entities(), 1, &device));
            let logits = logits.into_data().to_vec::<f32>().map_err(|e| format!("{e:?}"))?;
            let a = sample_action(&logits, &mut rng).0;
            let r = env.step_with(a, |emu| {
                let ram = emu.ram();
                xs.push(ram::u16_at(ram, ram::PLAYER_X));
                ys.push(ram::u16_at(ram, ram::PLAYER_Y));
            });
            if r.outcome != Outcome::Running {
                break r.outcome;
            }
        };
        let name = match outcome { Outcome::Cleared => "cleared", Outcome::Died => "died", _ => "timeout" };
        let join = |v: &[u16]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",");
        if ep > 0 {
            json.push(',');
        }
        write!(json, r#"{{"out":"{name}","cause":"{}","xs":[{}],"ys":[{}]}}"#, env.telemetry().cause.name(), join(&xs), join(&ys)).unwrap();
        println!("episode {ep}: {name}, {} frames, max_x {}", xs.len(), env.max_x());
    }
    json.push_str("]}");
    std::fs::write(cfg("OUT", format!("ghosts_L{id}.json")), json).map_err(|e| e.to_string())
}
