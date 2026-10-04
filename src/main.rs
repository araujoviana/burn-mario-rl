mod emulator;

use emulator::{Emulator, button};
use std::path::PathBuf;
use std::time::Instant;

/// Mario's X position in work RAM ($7E0094, 16 bit little endian).
fn mario_x(emu: &Emulator) -> u16 {
    let ram = emu.ram();
    u16::from_le_bytes([ram[0x94], ram[0x95]])
}

fn write_ppm(path: &str, rgb: &[u8], w: u32, h: u32) {
    let mut out = format!("P6\n{w} {h}\n255\n").into_bytes();
    out.extend_from_slice(rgb);
    std::fs::write(path, out).expect("write ppm");
}

fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let core = PathBuf::from(args.next().unwrap_or("/usr/lib/libretro/snes9x_libretro.so".into()));
    let rom = PathBuf::from(args.next().unwrap_or("Super Mario World (USA).sfc".into()));

    let mut emu = Emulator::load(&core, &rom)?;
    println!("core loaded, native fps {:.2}, work RAM {} bytes", emu.fps, emu.ram().len());

    // Boot and dump a frame every 300 frames to see what the game is showing.
    for i in 1..=2400u32 {
        emu.run_frame();
        if i % 300 == 0 {
            let (rgb, w, h) = emu.frame();
            let lit = rgb.iter().filter(|&&b| b > 16).count();
            println!("frame {i}: {w}x{h}, lit bytes {lit}");
            write_ppm(&format!("/tmp/smw_{i}.ppm"), &rgb, w, h);
        }
    }

    // Save state, run with RIGHT held, then restore and check RAM matches.
    let state = emu.save_state()?;
    println!("state size {} bytes", state.len());
    let ram_before = emu.ram().to_vec();

    emu.set_buttons(button::RIGHT);
    let frames = 6000u32;
    let start = Instant::now();
    for _ in 0..frames {
        emu.run_frame();
    }
    let secs = start.elapsed().as_secs_f64();
    println!("{frames} frames in {secs:.2}s = {:.0} fps ({:.1}x realtime)", frames as f64 / secs, frames as f64 / secs / emu.fps);
    println!("mario_x after run: {}", mario_x(&emu));

    emu.load_state(&state)?;
    let identical = emu.ram() == ram_before.as_slice();
    println!("RAM identical after load_state: {identical}");
    if !identical {
        return Err("save/load state round trip changed RAM".into());
    }
    Ok(())
}
