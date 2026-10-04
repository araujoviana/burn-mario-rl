//! Diagnostic: when does a level register as cleared? Env: LEVEL (id), LEVELS_DIR, CORE.
use burn_mario_rl::emulator::Emulator;
use burn_mario_rl::env::{ACTIONS, FRAME_SKIP};
use burn_mario_rl::levels::LevelSet;
use burn_mario_rl::ram;
use std::path::{Path, PathBuf};

fn main() -> Result<(), String> {
    let id: usize = std::env::var("LEVEL").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    let core = PathBuf::from(std::env::var("CORE").unwrap_or("/usr/lib/libretro/snes9x_libretro.so".into()));
    let levels = LevelSet::load(Path::new(&std::env::var("LEVELS_DIR").unwrap_or("levels".into())))?;
    let mut emu = Emulator::load(&core, &PathBuf::from("Super Mario World (USA).sfc"))?;
    let mut rng = 7u64;
    for attempt in 0..5 {
        emu.load_state(&levels.levels[id].state)?;
        println!("attempt {attempt}: at load: end_timer={} x={} mode={:#04x}", emu.ram()[ram::END_LEVEL_TIMER], ram::u16_at(emu.ram(), ram::PLAYER_X), emu.ram()[ram::GAME_MODE]);
        for step in 0..1500 {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let a = if rng >> 60 < 9 { 4 } else { (rng >> 33) as usize % ACTIONS.len() };
            emu.set_buttons(ACTIONS[a]);
            for _ in 0..FRAME_SKIP { emu.run_frame(); }
            let r = emu.ram();
            if r[ram::END_LEVEL_TIMER] != 0 {
                println!("   cleared at step {step}: x={} y={} end_timer={} timer_left={}", ram::u16_at(r, ram::PLAYER_X), ram::u16_at(r, ram::PLAYER_Y), r[ram::END_LEVEL_TIMER], ram::timer(r));
                break;
            }
            if r[ram::PLAYER_STATE] == 9 { println!("   died at step {step}"); break; }
        }
    }
    Ok(())
}
