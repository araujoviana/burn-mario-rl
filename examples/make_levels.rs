//! Scans the overworld map, enters every level tile, keeps the horizontal ones where a
//! biased-random policy makes progress, and writes them to LEVELS_DIR (default `levels`).
//! Env: CORE, LEVELS_DIR.
use burn_mario_rl::emulator::Emulator;
use burn_mario_rl::env::{ACTIONS, FRAME_SKIP, overworld_state};
use burn_mario_rl::levels::{Level, LevelSet, assign_split};
use burn_mario_rl::ram::{self, warp_to_position};
use std::path::PathBuf;

fn main() -> Result<(), String> {
    let core = PathBuf::from(std::env::var("CORE").unwrap_or("/usr/lib/libretro/snes9x_libretro.so".into()));
    let rom = PathBuf::from("Super Mario World (USA).sfc");
    let out = PathBuf::from(std::env::var("LEVELS_DIR").unwrap_or("levels".into()));
    let mut emu = Emulator::load(&core, &rom)?;
    let ow = overworld_state(&mut emu)?;

    // (translevel, x, y, state); the same level can be reached from one tile only, but guard anyway.
    let mut kept: Vec<(u8, u16, u16, Vec<u8>)> = Vec::new();
    let mut seen_tl = std::collections::HashSet::new();
    for ty in 0..32u16 {
        for tx in 0..32u16 {
            let (x, y) = (tx * 16 + 8, ty * 16 + 8);
            let state = match warp_to_position(&mut emu, &ow, x, y) {
                Ok(s) => s,
                Err(_) => continue, // not a level tile
            };
            let tl = emu.ram()[ram::TRANSLEVEL];
            if !seen_tl.insert(tl) {
                continue;
            }
            if ram::is_vertical(emu.ram()) {
                println!("({x:3},{y:3}) tl {tl:#04x}: skip (vertical)");
                continue;
            }
            // Progress probe on the same emulator (the core keeps global state, so a second
            // emulator in this process would crash): biased-random policy, best of 3 tries.
            let mut rng = 99u64 + tl as u64;
            let mut best = 0u16;
            for _ in 0..3 {
                emu.load_state(&state)?;
                for _ in 0..600 {
                    rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    let a = if rng >> 60 < 9 { 4 } else { (rng >> 33) as usize % ACTIONS.len() };
                    emu.set_buttons(ACTIONS[a]);
                    for _ in 0..FRAME_SKIP {
                        emu.run_frame();
                    }
                    best = best.max(ram::u16_at(emu.ram(), ram::PLAYER_X));
                    if emu.ram()[ram::PLAYER_STATE] == 9 || emu.ram()[ram::END_LEVEL_TIMER] != 0 {
                        break;
                    }
                }
            }
            emu.set_buttons(0);
            if best < 300 {
                println!("({x:3},{y:3}) tl {tl:#04x}: skip (random policy reached only x={best})");
                continue;
            }
            println!("({x:3},{y:3}) tl {tl:#04x}: keep (random policy reached x={best})");
            kept.push((tl, x, y, state));
        }
    }

    let split = assign_split(kept.len());
    let levels = kept
        .into_iter()
        .zip(split)
        .enumerate()
        .map(|(id, ((translevel, map_x, map_y, state), held_out))| Level { id, translevel, map_x, map_y, held_out, state })
        .collect();
    let set = LevelSet { levels };
    set.save(&out)?;
    println!("wrote {} levels ({} held out) to {}", set.levels.len(), set.held_out_ids().len(), out.display());
    Ok(())
}
