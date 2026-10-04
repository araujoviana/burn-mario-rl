//! Diagnostic: per level, saves the first frame and a late frame as PPM and reports how a
//! biased-random policy fares over many attempts. Env: LEVELS_DIR, OUT_DIR, CORE.
use burn_mario_rl::emulator::Emulator;
use burn_mario_rl::env::{ACTIONS, FRAME_SKIP};
use burn_mario_rl::levels::LevelSet;
use burn_mario_rl::ram;
use std::path::{Path, PathBuf};

fn write_ppm(path: &Path, emu: &Emulator) {
    let (rgb, w, h) = emu.frame();
    let mut out = format!("P6\n{w} {h}\n255\n").into_bytes();
    out.extend_from_slice(&rgb);
    std::fs::write(path, out).expect("write ppm");
}

fn main() -> Result<(), String> {
    let core = PathBuf::from(std::env::var("CORE").unwrap_or("/usr/lib/libretro/snes9x_libretro.so".into()));
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap_or("/tmp/levels_inspect".into()));
    std::fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    let levels = LevelSet::load(Path::new(&std::env::var("LEVELS_DIR").unwrap_or("levels".into())))?;
    let mut emu = Emulator::load(&core, &PathBuf::from("Super Mario World (USA).sfc"))?;
    let mut rng = 7u64;
    println!("id  tl   timer  attempts: clear/died/timeout  best_x  death_x(median)  max_x_ever_seen");
    for l in &levels.levels {
        emu.load_state(&l.state)?;
        emu.set_buttons(0);
        emu.run_frame();
        write_ppm(&out.join(format!("{:02}_start.ppm", l.id)), &emu);
        let timer = ram::timer(emu.ram());
        let (mut clear, mut died, mut timeout, mut mode_only) = (0, 0, 0, 0);
        let (mut best, mut death_xs, mut shot) = (0u16, Vec::new(), false);
        for _ in 0..20 {
            emu.load_state(&l.state)?;
            let (mut max_x, mut stall) = (0u16, 0);
            let mut outcome = 2;
            for step in 0..1500 {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let a = if rng >> 60 < 9 { 4 } else { (rng >> 33) as usize % ACTIONS.len() };
                emu.set_buttons(ACTIONS[a]);
                for _ in 0..FRAME_SKIP {
                    emu.run_frame();
                }
                let r = emu.ram();
                let x = ram::u16_at(r, ram::PLAYER_X);
                if x > max_x { max_x = x; stall = 0 } else { stall += 1 }
                if r[ram::PLAYER_STATE] == 9 { outcome = 1; break; }
                if r[ram::END_LEVEL_TIMER] != 0 || (r[ram::GAME_MODE] != 0x14 && r[ram::PLAYER_STATE] != 9) {
                    outcome = 0;
                    if r[ram::END_LEVEL_TIMER] == 0 {
                        mode_only += 1;
                        if mode_only == 1 { println!("    level {} false-clear? mode={:#04x} step={step} x={}", l.id, r[ram::GAME_MODE], x); }
                    }
                    break;
                }
                if stall >= 200 { break; }
                if !shot && step == 300 {
                    write_ppm(&out.join(format!("{:02}_mid.ppm", l.id)), &emu);
                    shot = true;
                }
            }
            match outcome { 0 => clear += 1, 1 => { died += 1; death_xs.push(max_x) } _ => timeout += 1 }
            best = best.max(max_x);
        }
        death_xs.sort();
        let med = death_xs.get(death_xs.len() / 2).copied().unwrap_or(0);
        println!("{:2}  {:#04x} {:4}   {:2}/{:2}/{:2} (of the clears, {mode_only} had no goal timer)  {:5}  {:5}", l.id, l.translevel, timer, clear, died, timeout, best, med);
    }
    Ok(())
}
