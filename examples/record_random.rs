//! Replays the biased-random policy until an episode is `Cleared`, then writes it to a video.
use burn_mario_rl::env::{ACTIONS, MarioEnv, Outcome};
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

fn main() -> Result<(), String> {
    let out = std::env::args().nth(1).unwrap_or("/tmp/random_clear.mp4".into());
    let state = std::fs::read("level1.state").map_err(|e| e.to_string())?;
    let mut env = MarioEnv::new(&PathBuf::from("/usr/lib/libretro/snes9x_libretro.so"), &PathBuf::from("Super Mario World (USA).sfc"), state)?;
    let mut rng = 12345u64;
    for ep in 0..40 {
        env.reset()?;
        let mut frames: Vec<Vec<u8>> = Vec::new();
        let outcome = loop {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let a = if rng >> 60 < 9 { 4 } else { (rng >> 33) as usize % ACTIONS.len() };
            let r = env.step(a);
            env.emulator().with_frame(|rgb, _, _| frames.push(rgb.to_vec()));
            if r.outcome != Outcome::Running { break r.outcome; }
        };
        println!("ep {ep}: {outcome:?}, {} frames", frames.len());
        if outcome == Outcome::Cleared {
            // Keep playing after the clear to confirm what the game does next.
            for i in 0..4u32 {
                for _ in 0..4 { env.step(0); }
                let mode = env.emulator().ram()[0x100];
                println!("  after clear +{} steps: mode {mode:#04x}", (i + 1) * 4);
            }
            for _ in 0..120 { env.step(0); }
            env.emulator().with_frame(|rgb, w, h| {
                let mut out = format!("P6\n{w} {h}\n255\n").into_bytes();
                out.extend_from_slice(rgb);
                std::fs::write("/tmp/after_clear.ppm", out).unwrap();
            });
            let mut ff = Command::new("ffmpeg")
                .args(["-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "256x224", "-r", "15", "-i", "-"])
                .args(["-vf", "scale=768:672:flags=neighbor", "-c:v", "libx264", "-pix_fmt", "yuv420p", &out])
                .stdin(Stdio::piped()).spawn().map_err(|e| e.to_string())?;
            let mut stdin = ff.stdin.take().unwrap();
            for f in &frames { stdin.write_all(f).map_err(|e| e.to_string())?; }
            drop(stdin);
            ff.wait().map_err(|e| e.to_string())?;
            println!("wrote {out}");
            return Ok(());
        }
    }
    Err("no cleared episode".into())
}
