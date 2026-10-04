//! Spike: checks RAM addresses and the cursor-teleport warp. Prints what it sees; the author reads it.
use burn_mario_rl::emulator::{Emulator, button};
use burn_mario_rl::env::overworld_state;
use burn_mario_rl::ram::{self, warp_to_position};
use std::path::PathBuf;

fn describe(emu: &Emulator) -> String {
    let r = emu.ram();
    format!(
        "mode={:#04x} tl={:#04x} x={:5} y={:5} timer={:3} vertical={} sprites={}",
        r[ram::GAME_MODE],
        r[ram::TRANSLEVEL],
        ram::u16_at(r, ram::PLAYER_X),
        ram::u16_at(r, ram::PLAYER_Y),
        ram::timer(r),
        ram::is_vertical(r),
        (0..ram::SPRITE_SLOTS).filter(|&i| r[ram::SPRITE_STATUS + i] != 0).count(),
    )
}

fn main() -> Result<(), String> {
    let core = PathBuf::from(std::env::var("CORE").unwrap_or("/usr/lib/libretro/snes9x_libretro.so".into()));
    let rom = PathBuf::from("Super Mario World (USA).sfc");
    let mut emu = Emulator::load(&core, &rom)?;
    let ow = overworld_state(&mut emu)?;
    let r = emu.ram();
    println!("cursor at ({}, {})", ram::u16_at(r, ram::MAP_X), ram::u16_at(r, ram::MAP_Y));

    // Warp to a few known level tiles (found by scanning the map; see make_levels).
    for (x, y) in [(152u16, 136u16), (264, 24), (56, 136), (8, 232)] {
        let state = warp_to_position(&mut emu, &ow, x, y)?;
        println!("warp ({x:3},{y:3}): {}", describe(&emu));
        if (x, y) == (152, 136) {
            // Timer tick length and sprite/Mario address checks in the first level.
            emu.load_state(&state)?;
            let (mut prev, mut since, mut ticks) = (ram::timer(emu.ram()), 0u32, Vec::new());
            emu.set_buttons(button::RIGHT | button::Y);
            for f in 0..600 {
                emu.run_frame();
                since += 1;
                let t = ram::timer(emu.ram());
                if t != prev {
                    ticks.push(since);
                    since = 0;
                    prev = t;
                }
                if f % 100 == 99 {
                    let r = emu.ram();
                    let sprites: Vec<String> = (0..ram::SPRITE_SLOTS)
                        .filter(|&i| r[ram::SPRITE_STATUS + i] != 0)
                        .map(|i| format!("#{i}:type {} st {} at ({},{})", r[ram::SPRITE_TYPE + i], r[ram::SPRITE_STATUS + i], r[ram::SPRITE_X_LO + i] as u16 | (r[ram::SPRITE_X_HI + i] as u16) << 8, r[ram::SPRITE_Y_LO + i] as u16 | (r[ram::SPRITE_Y_HI + i] as u16) << 8))
                        .collect();
                    println!(
                        "  f{f:3}: mario x={} y={} vx={} vy={} blocked={:#04x} power={} | {}",
                        ram::u16_at(r, ram::PLAYER_X),
                        ram::u16_at(r, ram::PLAYER_Y),
                        r[ram::PLAYER_SPEED_X] as i8,
                        r[ram::PLAYER_SPEED_Y] as i8,
                        r[ram::PLAYER_BLOCKED],
                        r[ram::POWERUP],
                        sprites.join("  "),
                    );
                }
            }
            println!("  frames between timer ticks: {ticks:?}");
        }
    }
    Ok(())
}
