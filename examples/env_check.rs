//! Smoke test: build the level-start state, run random episodes, report outcomes and speed.
use burn_mario_rl::emulator::Emulator;
use burn_mario_rl::env::{ACTIONS, MarioEnv, Outcome, level_start_state};
use std::path::PathBuf;
use std::time::Instant;

fn main() -> Result<(), String> {
    let core = PathBuf::from(std::env::var("CORE").unwrap_or("/usr/lib/libretro/snes9x_libretro.so".into()));
    let rom = PathBuf::from("Super Mario World (USA).sfc");
    let state = level_start_state(&mut Emulator::load(&core, &rom)?)?;
    std::fs::write("level1.state", &state).map_err(|e| e.to_string())?;
    println!("level-start state: {} bytes", state.len());

    let mut env = MarioEnv::new(&core, &rom, state)?;
    let mut rng = 12345u64;
    let (mut steps, start) = (0u64, Instant::now());
    for ep in 0..20 {
        env.reset()?;
        let (mut ret, mut n) = (0.0, 0);
        let outcome = loop {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let a = if rng >> 60 < 9 { 4 } else { (rng >> 33) as usize % ACTIONS.len() }; // biased to run+jump right
            let r = env.step(a);
            ret += r.reward;
            n += 1;
            if r.outcome != Outcome::Running { break r.outcome; }
        };
        steps += n;
        println!("ep {ep}: {outcome:?} after {n} steps, return {ret:.1}");
    }
    let secs = start.elapsed().as_secs_f64();
    println!("{steps} agent steps in {secs:.1}s = {:.0} steps/s on one core", steps as f64 / secs);
    Ok(())
}
