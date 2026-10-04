//! Scratch tool: boot, then run a button script given as args like `R:30 .:60 A:5`,
//! printing game mode, level id and X after each step. Dumps `/tmp/ex_<n>.ppm` per step.
use burn_mario_rl::emulator::{Emulator, button};
use std::path::PathBuf;

fn mask(name: &str) -> u16 {
    name.chars().map(|c| match c {
        'R' => button::RIGHT, 'L' => button::LEFT, 'U' => button::UP, 'D' => button::DOWN,
        'A' => button::A, 'B' => button::B, 'X' => button::X, 'Y' => button::Y,
        'S' => button::START, _ => 0,
    }).fold(0, |a, b| a | b)
}

fn main() -> Result<(), String> {
    let mut emu = Emulator::load(&PathBuf::from(std::env::var("CORE").unwrap_or("/usr/lib/libretro/snes9x_libretro.so".into())), &PathBuf::from("Super Mario World (USA).sfc"))?;
    let report = |emu: &Emulator, tag: &str| {
        let r = emu.ram();
        println!("{tag}: mode={:#04x} level={:#04x} x={} lives={}", r[0x100], r[0x13BF], u16::from_le_bytes([r[0x94], r[0x95]]), r[0xDBE]);
    };
    // Boot: wait out the logo, then tap START until the game leaves the title/intro.
    for f in 0..2400u32 {
        emu.set_buttons(if f > 600 && f % 90 < 5 { button::START } else { 0 });
        emu.run_frame();
    }
    report(&emu, "after boot");
    for (i, step) in std::env::args().skip(1).enumerate() {
        let (keys, frames) = step.split_once(':').ok_or("bad step")?;
        emu.set_buttons(mask(keys));
        for _ in 0..frames.parse::<u32>().map_err(|e| e.to_string())? { emu.run_frame(); }
        report(&emu, &step);
        let (rgb, w, h) = emu.frame();
        let mut out = format!("P6\n{w} {h}\n255\n").into_bytes();
        out.extend_from_slice(&rgb);
        std::fs::write(format!("/tmp/ex_{i}.ppm"), out).unwrap();
    }
    Ok(())
}
