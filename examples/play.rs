//! Play a level yourself and record the run as a demonstration for training.
//!
//!   cargo run --release --features play --example play        (env: LEVEL=<id>, CORE=<path to core>)
//!
//! Keys: arrows = d-pad, X = jump (B), Z = run (Y), S = spin jump (A), R = restart the level, Esc = quit.
//! Dying restarts the level automatically. When you clear it the run is saved to
//! `demos/level<id>_<frames>f.demo` (one joypad mask per frame, comma separated) and the tool exits.
//! Training replays these files to build a winning path (see `vec_env::load_saved_wins`).
use burn_mario_rl::emulator::{Emulator, button};
use burn_mario_rl::levels::LevelSet;
use burn_mario_rl::ram;
use minifb::{Key, Window, WindowOptions};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

const SCALE: usize = 3;

fn cfg<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn pad(window: &Window) -> u16 {
    let mut m = 0;
    for (key, bit) in [
        (Key::Left, button::LEFT),
        (Key::Right, button::RIGHT),
        (Key::Up, button::UP),
        (Key::Down, button::DOWN),
        (Key::X, button::B),
        (Key::Z, button::Y),
        (Key::S, button::A),
    ] {
        if window.is_key_down(key) {
            m |= bit;
        }
    }
    m
}

fn draw(emu: &Emulator, window: &mut Window, buf: &mut Vec<u32>) {
    emu.with_frame(|rgb, w, h| {
        let (w, h) = (w as usize, h as usize);
        buf.resize(w * SCALE * h * SCALE, 0);
        for y in 0..h * SCALE {
            for x in 0..w * SCALE {
                let p = &rgb[((y / SCALE) * w + x / SCALE) * 3..][..3];
                buf[y * w * SCALE + x] = (p[0] as u32) << 16 | (p[1] as u32) << 8 | p[2] as u32;
            }
        }
        let _ = window.update_with_buffer(buf, w * SCALE, h * SCALE);
    });
}

fn main() -> Result<(), String> {
    let core = PathBuf::from(cfg("CORE", "/usr/lib/libretro/snes9x_libretro.so".to_string()));
    let rom = PathBuf::from(cfg("ROM", "Super Mario World (USA).sfc".to_string()));
    let levels = LevelSet::load(Path::new(&cfg("LEVELS", "levels".to_string())))?;
    let id: usize = cfg("LEVEL", 3);
    let level = levels.levels.iter().find(|l| l.id == id).ok_or(format!("no level with id {id}"))?;
    let mut emu = Emulator::load(&core, &rom)?;
    let mut window = Window::new(&format!("level {id}  |  arrows move, X jump, Z run, S spin, R restart, Esc quit"), 256 * SCALE, 224 * SCALE, WindowOptions::default()).map_err(|e| e.to_string())?;
    let mut buf = Vec::new();
    let frame_time = Duration::from_micros(16_667);
    let mut attempt = 0;
    'attempts: loop {
        attempt += 1;
        emu.load_state(&level.state)?;
        emu.set_buttons(0);
        emu.run_frame();
        window.set_title(&format!("level {id}, attempt {attempt}  |  arrows move, X jump, Z run, S spin, R restart, Esc quit"));
        let mut masks: Vec<u16> = Vec::new();
        let mut next = Instant::now();
        // a short pause so you can get your hands ready
        for _ in 0..90 {
            draw(&emu, &mut window, &mut buf);
            next += frame_time;
            std::thread::sleep(next.saturating_duration_since(Instant::now()));
        }
        let mut after_end = 0;
        loop {
            if !window.is_open() || window.is_key_down(Key::Escape) {
                println!("quit without saving");
                return Ok(());
            }
            if window.is_key_down(Key::R) {
                continue 'attempts;
            }
            let mask = pad(&window);
            masks.push(mask);
            emu.set_buttons(mask);
            emu.run_frame();
            draw(&emu, &mut window, &mut buf);
            let r = emu.ram();
            let (dead, cleared) = (r[ram::PLAYER_STATE] == 9, r[ram::END_LEVEL_TIMER] != 0);
            if dead || cleared {
                after_end += 1;
                if after_end > 90 {
                    if cleared {
                        std::fs::create_dir_all("demos").map_err(|e| e.to_string())?;
                        let path = format!("demos/level{id}_{}f.demo", masks.len());
                        let text: Vec<String> = masks.iter().map(|m| m.to_string()).collect();
                        std::fs::write(&path, text.join(",")).map_err(|e| e.to_string())?;
                        println!("cleared! saved {path} ({} frames)", masks.len());
                        return Ok(());
                    }
                    println!("died on attempt {attempt}, restarting");
                    continue 'attempts;
                }
            }
            next += frame_time;
            std::thread::sleep(next.saturating_duration_since(Instant::now()));
        }
    }
}
