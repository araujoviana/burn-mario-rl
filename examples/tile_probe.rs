//! Prints the Map16 tile ids around Mario at the start of each level, to check the RAM tilemap layout.
use burn_mario_rl::emulator::Emulator;
use burn_mario_rl::levels::LevelSet;
use burn_mario_rl::ram;
use std::path::{Path, PathBuf};

fn main() -> Result<(), String> {
    let core = PathBuf::from(std::env::var("CORE").unwrap_or("/usr/lib/libretro/snes9x_libretro.so".into()));
    let ids: Vec<usize> = std::env::var("IDS").unwrap_or("7,0,5".into()).split(',').filter_map(|v| v.parse().ok()).collect();
    let levels = LevelSet::load(Path::new("levels"))?;
    let mut emu = Emulator::load(&core, &PathBuf::from("Super Mario World (USA).sfc"))?;
    for l in levels.levels.iter().filter(|l| ids.contains(&l.id)) {
        emu.load_state(&l.state)?;
        emu.set_buttons(0);
        for _ in 0..30 { emu.run_frame(); }
        let r = emu.ram();
        println!("ram len {}", r.len());
        let (mx, my) = (ram::u16_at(r, ram::PLAYER_X) as usize, ram::u16_at(r, ram::PLAYER_Y) as usize);
        println!("== level {} (tl {:#x}) mario x {mx} y {my} vertical {}", l.id, l.translevel, ram::is_vertical(r));
        let (cx, cy) = (mx / 16, my / 16);
        for ty in cy.saturating_sub(10)..cy + 5 {
            let mut line = String::new();
            for tx in cx.saturating_sub(6)..cx + 20 {
                // horizontal layout: 16x27 tiles per screen
                let off = (tx / 16) * 0x1B0 + ty * 16 + (tx % 16);
                let lo = r[0xC800 + off] as usize;
                let hi = r[0x1C800 + off] as usize;
                line.push_str(&format!("{:3x}", hi << 8 | lo));
            }
            println!("{ty:2}|{line}");
        }
    }
    Ok(())
}
