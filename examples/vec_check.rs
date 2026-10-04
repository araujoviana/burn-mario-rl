//! Smoke test: multi-level VecEnv with a uniform sampler, then the door-level clear rule.
//! Env: ENVS (4), STEPS (1500), CORE, LEVELS_DIR.
use burn_mario_rl::env::{ACTIONS, MarioEnv, Outcome};
use burn_mario_rl::levels::LevelSet;
use burn_mario_rl::plr::{LevelSampler, PlrConfig};
use burn_mario_rl::vec_env::VecEnv;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

fn cfg<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn main() -> Result<(), String> {
    let core = PathBuf::from(cfg("CORE", "/usr/lib/libretro/snes9x_libretro.so".to_string()));
    let rom = PathBuf::from("Super Mario World (USA).sfc");
    let levels = Arc::new(LevelSet::load(Path::new(&cfg("LEVELS_DIR", "levels".to_string())))?);
    let all: Vec<usize> = levels.levels.iter().map(|l| l.id).collect();
    let (n, steps): (usize, usize) = (cfg("ENVS", 4), cfg("STEPS", 1500));

    let sampler = LevelSampler::shared(all, PlrConfig { enabled: false, ..Default::default() });
    let mut envs = VecEnv::new(n, &core, &rom, levels.clone(), sampler, 1.0, 1)?;
    println!("{n} envs started on levels {:?}", envs.levels);
    let mut per_level: BTreeMap<usize, [u32; 4]> = BTreeMap::new(); // died, cleared, timeout, episodes
    let (mut rng, start) = (99u64, Instant::now());
    let mut reward_sum = 0.0f32;
    for _ in 0..steps {
        let actions: Vec<usize> = (0..n)
            .map(|_| {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                if rng >> 60 < 9 { 4 } else { (rng >> 33) as usize % ACTIONS.len() }
            })
            .collect();
        let batch = envs.step(&actions);
        reward_sum += batch.rewards.iter().sum::<f32>();
        assert!(batch.step_levels.iter().all(|&l| l < levels.levels.len()));
        for e in batch.finished {
            let c = per_level.entry(e.level).or_default();
            c[3] += 1;
            match e.outcome {
                Outcome::Died => c[0] += 1,
                Outcome::Cleared => c[1] += 1,
                _ => c[2] += 1,
            }
        }
    }
    let secs = start.elapsed().as_secs_f32();
    println!("{} agent steps in {secs:.1}s = {:.0} steps/s with {n} envs, reward sum {reward_sum:.0}", steps * n, (steps * n) as f32 / secs);
    println!("levels seen: {} of {}", per_level.len(), levels.levels.len());
    for (l, c) in &per_level {
        println!("  level {l:2}: {} episodes: {} died, {} cleared, {} timeout", c[3], c[0], c[1], c[2]);
    }
    drop(envs);

    // Door levels (tl 0x3f, 0x45): the old rule called entering the door a clear at x~736.
    for tl in [0x3fu8, 0x45] {
        let id = levels.levels.iter().find(|l| l.translevel == tl).ok_or("door level missing")?.id;
        let mut env = MarioEnv::with_levels(&core, &rom, levels.clone(), 1.0)?;
        let (mut doors, mut false_clears, mut past_door, mut real) = (0, 0, 0, 0);
        let mut rng = 5u64;
        for _ in 0..30 {
            env.reset_to(id)?;
            let mut prev_x = 0u16;
            for _ in 0..1500 {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let a = if rng >> 60 < 9 { 4 } else { (rng >> 33) as usize % ACTIONS.len() };
                let r = env.step(a);
                let x = env.x();
                if prev_x > 700 && x + 200 < prev_x {
                    doors += 1; // x jumped back: a new area
                }
                if doors > 0 && x > 100 { past_door += 1; }
                prev_x = x;
                if r.outcome == Outcome::Cleared { real += 1; }
                if r.outcome != Outcome::Running { break; }
            }
            let _ = &mut false_clears;
        }
        println!("door level tl {tl:#04x} (id {id}): doors taken {doors}, steps spent in the next area {past_door}, clears {real}");
    }
    Ok(())
}
