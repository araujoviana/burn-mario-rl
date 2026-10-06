//! Replay every demo in `$DEMOS_DIR` (default `demos`) from its level start and report whether it clears.
use burn_mario_rl::env::MarioEnv;
use burn_mario_rl::levels::LevelSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

fn main() -> Result<(), String> {
    let core = PathBuf::from(std::env::var("CORE").unwrap_or("/usr/lib/libretro/snes9x_libretro.so".into()));
    let levels = Arc::new(LevelSet::load(Path::new("levels"))?);
    let mut env = MarioEnv::with_levels(&core, &PathBuf::from("Super Mario World (USA).sfc"), levels, 1.0)?;
    let dir = std::env::var("DEMOS_DIR").unwrap_or("demos".into());
    for e in std::fs::read_dir(&dir).map_err(|e| e.to_string())?.flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        let Some(rest) = name.strip_prefix("level").and_then(|r| r.strip_suffix("f.demo")) else { continue };
        let Some((id, _)) = rest.split_once('_') else { continue };
        let Some(level) = id.parse::<usize>().ok().and_then(|id| env.level_index(id)) else { continue };
        let text = std::fs::read_to_string(e.path()).map_err(|e| e.to_string())?;
        let masks: Vec<u16> = text.trim().split(',').filter_map(|v| v.trim().parse().ok()).collect();
        let res = env.replay_masks(level, &masks, 64);
        println!("{name}: {} frames, {}", masks.len(), if res.is_some() { "CLEARS" } else { "does NOT clear" });
    }
    std::process::exit(0);
}
