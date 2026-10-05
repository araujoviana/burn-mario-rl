//! Checks the coin counter address ($0DBF): plays a biased-random policy on each level and prints when it changes.
use burn_mario_rl::emulator::Emulator;
use burn_mario_rl::env::{ACTIONS, FRAME_SKIP};
use burn_mario_rl::levels::LevelSet;
use std::path::{Path, PathBuf};

fn main() -> Result<(), String> {
    let core = PathBuf::from(std::env::var("CORE").unwrap_or("/usr/lib/libretro/snes9x_libretro.so".into()));
    let levels = LevelSet::load(Path::new("levels"))?;
    let mut emu = Emulator::load(&core, &PathBuf::from("Super Mario World (USA).sfc"))?;
    let mut rng = 11u64;
    for l in &levels.levels {
        let mut seen = Vec::new();
        for _ in 0..6 {
            emu.load_state(&l.state)?;
            let mut last = emu.ram()[0xDBF];
            for _ in 0..1200 {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let a = if rng >> 60 < 9 { 4 } else { (rng >> 33) as usize % ACTIONS.len() };
                emu.set_buttons(ACTIONS[a]);
                for _ in 0..FRAME_SKIP { emu.run_frame(); }
                let c = emu.ram()[0xDBF];
                if c != last { seen.push(c); last = c; }
                if emu.ram()[0x71] == 9 { break; }
            }
        }
        println!("level {:2}: start coins {} changes {:?}", l.id, 0, &seen[..seen.len().min(12)]);
    }
    Ok(())
}
