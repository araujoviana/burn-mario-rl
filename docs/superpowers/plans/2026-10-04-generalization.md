# Generalization Across Levels Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train one agent that clears many Super Mario World levels (including held-out ones) and clears them fast, using a multi-level env, PLR level sampling, an IMPALA trunk, entity RAM input and DrAC augmentation.

**Architecture:** Workers reset into a level chosen per episode by a shared `LevelSampler` (PLR or uniform). Observations become stacked frames plus an entity vector read from RAM; the network is a selectable trunk (IMPALA or Nature) with an entity MLP fused before the actor/critic heads. Every new piece sits behind an env-var flag in `train.rs`.

**Tech Stack:** Rust 2024, Burn 0.21 (ndarray locally, cuda on the VM), hand-written libretro host (snes9x), ffmpeg for video.

**Spec:** `docs/superpowers/specs/2026-10-04-generalization-design.md`

## Global Constraints

- Level pool: 15+ levels, 3-4 held out for a generalization score. If fewer than 15 pass the filter in Task 2, keep going and report the real number.
- Level start states come from a RAM warp from the overworld, not per-level menu scripting.
- Entity RAM enters the network as a vector branch fused with the CNN features (not extra image channels).
- Every new feature is behind a config flag so it can be ablated. PLR off = uniform sampling.
- Pool is limited to horizontal levels with a goal gate (layout flag filter).
- Step penalty stays `0.02`; progress reward stays `dx / 16`; death penalty stays `20`.
- Clear bonus is `100 * (1 + k * timer_left_fraction)` with `k` in config (default 1.0).
- Per-level step budget derives from the in-game timer, replacing the fixed `MAX_STEPS = 1500`.
- Frame skip 4, frame stack 4, grayscale 84x84 (unchanged).
- Backend stays a generic parameter; `cargo test` runs on the default ndarray backend with no ROM.
- The ROM (`Super Mario World (USA).sfc`) and `*.state` files are never committed (already gitignored). Start states for levels live in `levels/` which must be added to `.gitignore`.
- Old checkpoints (`vmD_best`) are incompatible after Task 7; do not try to keep them loadable.
- Do not start, stop or delete cloud VMs. A VM training run needs the user's explicit OK (see Task 11).
- Rust style: match the surrounding code (short doc comments, no heavy commenting, `cfg(name, default)` env helper pattern in binaries).

## File Structure

| File | Responsibility |
|---|---|
| `src/rng.rs` (new) | xorshift helper shared by new code |
| `src/ram.rs` (new) | All SMW RAM addresses and decoding helpers (timer, layout flag, sprites, Mario) |
| `src/entities.rs` (new) | `entity_vector(ram) -> [f32; ENT_LEN]` |
| `src/levels.rs` (new) | `Level`, `LevelSet`, manifest read/write, held-out split |
| `src/plr.rs` (new) | `LevelSampler`, PLR math, `level_scores` |
| `src/augment.rs` (new) | random-shift augmentation on observation bytes |
| `src/reward.rs` (new) | pure reward functions: clear bonus, step cap |
| `src/emulator.rs` | add `write_ram` |
| `src/env.rs` | multi-level reset, timer, entity vector, new reward wiring |
| `src/vec_env.rs` | shared sampler, level id per step, entity buffer |
| `src/ppo.rs` | `NetConfig`, trunks, entity branch, DrAC loss |
| `src/bin/train.rs` | flags, rollout buffers, PLR updates, held-out eval, logging |
| `src/bin/eval.rs` | level selection, clear time |
| `examples/warp_probe.rs` (new) | Task 1 spike |
| `examples/make_levels.rs` (new) | builds `levels/` from the overworld |
| `examples/make_curriculum.rs`, `examples/bench_net.rs`, `examples/env_check.rs` | adapt to new signatures |

---

### Task 1: Spike - RAM addresses and the level warp

This is exploratory. Its deliverable is `src/ram.rs` with verified addresses and a working `warp_to_level` function. Later tasks depend on it. If the warp cannot be made to work after a serious attempt, STOP and report to the user (fallback in the spec: scripted overworld walks, fewer levels).

**Files:**
- Modify: `src/emulator.rs` (add `write_ram`)
- Modify: `src/env.rs` (split `level_start_state`; add `overworld_state`)
- Create: `src/ram.rs`, `examples/warp_probe.rs`
- Modify: `src/lib.rs`

**Interfaces:**
- Produces: `Emulator::write_ram(&mut self, offset: usize, data: &[u8])`
- Produces: `env::overworld_state(emu: &mut Emulator) -> Result<Vec<u8>, String>` (save state on the overworld right after leaving Yoshi's House, before entering level 1)
- Produces: `ram::warp_to_level(emu: &mut Emulator, overworld: &[u8], translevel: u8) -> Result<Vec<u8>, String>` (returns a save state at the first controllable frame of that level)
- Produces constants in `ram.rs` used by later tasks (see Step 4).

- [ ] **Step 1: Add `write_ram` to the emulator**

In `src/emulator.rs`, after `ram()`:

```rust
    /// Overwrite work RAM at `offset` (`$7E0000 + offset`).
    pub fn write_ram(&mut self, offset: usize, data: &[u8]) {
        unsafe {
            let ptr = (self.get_memory_data)(MEMORY_SYSTEM_RAM) as *mut u8;
            let len = (self.get_memory_size)(MEMORY_SYSTEM_RAM);
            assert!(!ptr.is_null() && offset + data.len() <= len, "write_ram out of range");
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr.add(offset), data.len());
        }
    }
```

- [ ] **Step 2: Split the boot script**

In `src/env.rs`, replace `level_start_state` with two functions. `overworld_state` runs the existing script up to and including `hold(emu, 0, 200);` that follows "walk to the first level" (stop before the final `drive(... MODE_IN_LEVEL)`), then returns `emu.save_state()`. `level_start_state` calls the same script and finishes as before. Concretely:

```rust
/// Boot to the overworld just outside Yoshi's House, standing before the first level.
fn boot_to_overworld(emu: &mut Emulator) -> Result<(), String> {
    hold(emu, 0, 600); // ROM intro is black for ~5 s
    drive(emu, button::START, true, 3000, |e| game_mode(e) == MODE_OVERWORLD)?; // title -> intro -> map
    hold(emu, 0, 60);
    drive(emu, button::A, true, 600, |e| game_mode(e) == MODE_IN_LEVEL)?; // enter Yoshi's House
    hold(emu, 0, 30);
    drive(emu, button::RIGHT, false, 1200, |e| game_mode(e) == MODE_OVERWORLD)?; // walk out of the house
    hold(emu, 0, 300); // path to the next node opens
    hold(emu, button::RIGHT, 40); // walk to the first level
    hold(emu, 0, 200);
    Ok(())
}

/// Save state on the overworld, before the first level is entered. Source for `ram::warp_to_level`.
pub fn overworld_state(emu: &mut Emulator) -> Result<Vec<u8>, String> {
    boot_to_overworld(emu)?;
    emu.save_state()
}

/// Play the menus from power-on to the first frame of the first level and return a save state there.
pub fn level_start_state(emu: &mut Emulator) -> Result<Vec<u8>, String> {
    boot_to_overworld(emu)?;
    drive(emu, button::A, true, 600, |e| game_mode(e) == MODE_IN_LEVEL)?;
    hold(emu, 0, 60); // let the level fade in and Mario land
    emu.save_state()
}
```

Make `game_mode`, `hold` and `drive` `pub(crate)`.

- [ ] **Step 3: Create `src/ram.rs` with candidate addresses and the probe**

```rust
//! Super Mario World work-RAM addresses (offsets from $7E0000) and decoding helpers.
//! Every address here was checked with `examples/warp_probe.rs`; see the comments for how.

use crate::emulator::Emulator;
use crate::env::{drive_to_level, game_mode};

pub const GAME_MODE: usize = 0x100;
pub const PLAYER_STATE: usize = 0x71;
pub const PLAYER_X: usize = 0x94; // 16 bit
pub const PLAYER_Y: usize = 0x96; // 16 bit
pub const PLAYER_SPEED_X: usize = 0x7B; // signed
pub const PLAYER_SPEED_Y: usize = 0x7D; // signed
pub const PLAYER_BLOCKED: usize = 0x77; // bit 2 set = standing on something
pub const POWERUP: usize = 0x19;
pub const TRANSLEVEL: usize = 0x13BF;
pub const END_LEVEL_TIMER: usize = 0x1493;
pub const LAYOUT_FLAGS: usize = 0x5B; // bit 0 set = vertical level
pub const TIMER_HUNDREDS: usize = 0xF31;
pub const TIMER_TENS: usize = 0xF32;
pub const TIMER_ONES: usize = 0xF33;

pub const SPRITE_SLOTS: usize = 12;
pub const SPRITE_STATUS: usize = 0x14C8; // 0 = empty slot
pub const SPRITE_TYPE: usize = 0x9E;
pub const SPRITE_X_LO: usize = 0xE4;
pub const SPRITE_X_HI: usize = 0x14E0;
pub const SPRITE_Y_LO: usize = 0xD8;
pub const SPRITE_Y_HI: usize = 0x14D4;

pub fn u16_at(ram: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([ram[offset], ram[offset + 1]])
}

/// In-game countdown timer, 0..=999.
pub fn timer(ram: &[u8]) -> u32 {
    ram[TIMER_HUNDREDS] as u32 * 100 + ram[TIMER_TENS] as u32 * 10 + ram[TIMER_ONES] as u32
}

pub fn is_vertical(ram: &[u8]) -> bool {
    ram[LAYOUT_FLAGS] & 1 != 0
}
```

Add `pub mod ram;` to `src/lib.rs`. In `src/env.rs` add a `pub(crate) fn drive_to_level(emu: &mut Emulator, max: u32) -> Result<(), String>` that is `drive(emu, button::A, true, max, |e| game_mode(e) == MODE_IN_LEVEL)`.

Write `warp_to_level` in `ram.rs` (first attempt; Step 5 iterates on it):

```rust
/// Load `overworld`, force-enter `translevel`, and return a save state on the level's first
/// controllable frame. Errors if the level never reaches in-level mode.
pub fn warp_to_level(emu: &mut Emulator, overworld: &[u8], translevel: u8) -> Result<Vec<u8>, String> {
    emu.load_state(overworld)?;
    emu.set_buttons(0);
    emu.run_frame();
    emu.write_ram(TRANSLEVEL, &[translevel]);
    // Pressing A on a level tile makes the game start loading the level; mimic that.
    drive_to_level(emu, 600)?;
    for _ in 0..60 {
        emu.run_frame(); // fade in, Mario lands
    }
    if game_mode(emu) != 0x14 {
        return Err(format!("translevel {translevel:#04x}: ended in mode {:#04x}", game_mode(emu)));
    }
    emu.save_state()
}
```

Create `examples/warp_probe.rs`:

```rust
//! Spike: checks RAM addresses and whether writing the translevel number from the overworld
//! loads the requested level. Prints a table; no pass/fail, the author reads it.
use burn_mario_rl::emulator::Emulator;
use burn_mario_rl::env::overworld_state;
use burn_mario_rl::ram::{self, warp_to_level};
use std::path::PathBuf;

fn main() -> Result<(), String> {
    let core = PathBuf::from(std::env::var("CORE").unwrap_or("/usr/lib/libretro/snes9x_libretro.so".into()));
    let rom = PathBuf::from("Super Mario World (USA).sfc");
    let mut emu = Emulator::load(&core, &rom)?;
    let ow = overworld_state(&mut emu)?;

    // 1. Baseline: warping to the translevel we already reach by hand must match the hand-made state.
    println!("translevel_at_overworld = {:#04x}", emu.ram()[ram::TRANSLEVEL]);

    // 2. Sweep every translevel number.
    for tl in 0u8..=0x24 {
        match warp_to_level(&mut emu, &ow, tl) {
            Ok(_) => {
                let r = emu.ram();
                println!(
                    "tl {tl:#04x}: ok  x={:5} y={:5} timer={:3} vertical={} sprites_active={}",
                    ram::u16_at(r, ram::PLAYER_X),
                    ram::u16_at(r, ram::PLAYER_Y),
                    ram::timer(r),
                    ram::is_vertical(r),
                    (0..ram::SPRITE_SLOTS).filter(|&i| r[ram::SPRITE_STATUS + i] != 0).count(),
                );
            }
            Err(e) => println!("tl {tl:#04x}: FAIL {e}"),
        }
    }

    // 3. Timer tick length: frames per one timer decrement in level 1.
    let l1 = warp_to_level(&mut emu, &ow, emu_level1_translevel(&mut emu, &ow))?;
    emu.load_state(&l1)?;
    let (mut last, mut since, mut ticks) = (ram::timer(emu.ram()), 0u32, Vec::new());
    for _ in 0..600 {
        emu.run_frame();
        since += 1;
        let t = ram::timer(emu.ram());
        if t != last {
            ticks.push(since);
            since = 0;
            last = t;
        }
    }
    println!("frames between timer ticks: {ticks:?}");
    Ok(())
}

/// The translevel the overworld cursor currently points at.
fn emu_level1_translevel(emu: &mut Emulator, _ow: &[u8]) -> u8 {
    emu.ram()[ram::TRANSLEVEL]
}
```

- [ ] **Step 4: Run the probe**

Run: `cargo run --release --example warp_probe 2>&1 | tail -60`
Expected: a table. Success means: for at least 15 translevel numbers the row says `ok` with `vertical=false`, `timer` a plausible 100-500 value, and the frames-between-ticks list is constant (all equal, note the value).

If every row says FAIL: the game-mode trigger is wrong. In `warp_to_level`, instead of `drive_to_level` after writing `TRANSLEVEL`, write the game mode byte (`emu.write_ram(ram::GAME_MODE, &[m])`) for each `m` in `[0x0F, 0x10, 0x11, 0x12]` (reload the overworld state between attempts) and keep the one that reaches mode `0x14`. Also try also setting `$0DD6`-adjacent overworld cursor bytes only if needed. Record which trigger worked in a comment on `warp_to_level`.

If rows load but the wrong level (check by comparing: warping to level 1's translevel must reproduce `level1.state`'s first-frame `PLAYER_X`, `PLAYER_Y` and `timer`): the translevel write is taking effect after the loader reads it; write it in a frame earlier and re-test.

- [ ] **Step 5: Confirm sprite and Mario addresses**

Edit `examples/warp_probe.rs` to also run 300 frames of RIGHT held in a level that has sprites, printing sprite slots with `status != 0` as `(slot, type, x, y)` each 60 frames. Confirm sprite X values change as the screen scrolls, and that `PLAYER_SPEED_X` is positive and growing while running right, `PLAYER_BLOCKED & 4` is set while standing. If any address is wrong, correct the constant in `ram.rs` (comment how it was verified).

- [ ] **Step 6: Record measured constants**

Add to `src/ram.rs`, using the measured value from Step 4:

```rust
/// Emulator frames per one tick of the in-game timer (measured by `examples/warp_probe.rs`).
pub const TIMER_TICK_FRAMES: u32 = 40;
```

Replace `40` with the measured constant. Add `levels/` to `.gitignore`.

- [ ] **Step 7: Commit**

```bash
cargo check --examples
git add src examples .gitignore
git commit -m "Add RAM address module, write_ram and level warp probe"
```

---

### Task 2: LevelSet and `make_levels`

**Files:**
- Create: `src/levels.rs`, `examples/make_levels.rs`
- Modify: `src/lib.rs` (`pub mod levels;`)

**Interfaces:**
- Consumes: `ram::warp_to_level`, `env::overworld_state`, `ram::is_vertical`
- Produces:
  - `pub struct Level { pub id: usize, pub translevel: u8, pub held_out: bool, pub state: Vec<u8> }`
  - `pub struct LevelSet { pub levels: Vec<Level> }`
  - `LevelSet::load(dir: &Path) -> Result<LevelSet, String>`
  - `LevelSet::save(&self, dir: &Path) -> Result<(), String>`
  - `LevelSet::train_ids(&self) -> Vec<usize>`, `LevelSet::held_out_ids(&self) -> Vec<usize>`
  - `pub fn assign_split(count: usize) -> Vec<bool>` (true = held out)

- [ ] **Step 1: Write the failing tests**

In `src/levels.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_holds_out_every_fifth_capped_at_four() {
        assert_eq!(assign_split(3), vec![false, false, false]);
        let s = assign_split(15);
        assert_eq!(s.iter().filter(|&&h| h).count(), 3);
        assert!(s[4] && s[9] && s[14]);
        let big = assign_split(40);
        assert_eq!(big.iter().filter(|&&h| h).count(), 4);
    }

    #[test]
    fn manifest_round_trip() {
        let dir = std::env::temp_dir().join(format!("levels_test_{}", std::process::id()));
        let set = LevelSet {
            levels: vec![
                Level { id: 0, translevel: 5, held_out: false, state: vec![1, 2, 3] },
                Level { id: 1, translevel: 9, held_out: true, state: vec![4, 5] },
            ],
        };
        set.save(&dir).unwrap();
        let back = LevelSet::load(&dir).unwrap();
        assert_eq!(back.levels.len(), 2);
        assert_eq!(back.levels[1].translevel, 9);
        assert!(back.levels[1].held_out);
        assert_eq!(back.levels[1].state, vec![4, 5]);
        assert_eq!(back.train_ids(), vec![0]);
        assert_eq!(back.held_out_ids(), vec![1]);
        let _ = std::fs::remove_dir_all(dir);
    }
}
```

- [ ] **Step 2: Run to verify they fail**

Run: `cargo test --lib levels`
Expected: compile error (module missing).

- [ ] **Step 3: Implement**

```rust
//! The set of levels the agent trains and is tested on, stored as `manifest.txt` plus one
//! `<id>.state` file per level.

use std::path::Path;

pub struct Level {
    pub id: usize,
    pub translevel: u8,
    pub held_out: bool,
    pub state: Vec<u8>,
}

pub struct LevelSet {
    pub levels: Vec<Level>,
}

/// Every fifth level is held out (index 4, 9, ...), at most four, none when there are fewer than five.
pub fn assign_split(count: usize) -> Vec<bool> {
    let mut held = 0;
    (0..count)
        .map(|i| {
            let hold = i % 5 == 4 && held < 4;
            held += hold as usize;
            hold
        })
        .collect()
}

impl LevelSet {
    pub fn train_ids(&self) -> Vec<usize> {
        self.levels.iter().filter(|l| !l.held_out).map(|l| l.id).collect()
    }

    pub fn held_out_ids(&self) -> Vec<usize> {
        self.levels.iter().filter(|l| l.held_out).map(|l| l.id).collect()
    }

    pub fn save(&self, dir: &Path) -> Result<(), String> {
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        let mut manifest = String::new();
        for l in &self.levels {
            manifest.push_str(&format!("{} {} {}\n", l.id, l.translevel, if l.held_out { "held_out" } else { "train" }));
            std::fs::write(dir.join(format!("{}.state", l.id)), &l.state).map_err(|e| e.to_string())?;
        }
        std::fs::write(dir.join("manifest.txt"), manifest).map_err(|e| e.to_string())
    }

    pub fn load(dir: &Path) -> Result<Self, String> {
        let manifest = std::fs::read_to_string(dir.join("manifest.txt")).map_err(|e| format!("{}: {e} (run make_levels)", dir.display()))?;
        let mut levels = Vec::new();
        for line in manifest.lines().filter(|l| !l.trim().is_empty()) {
            let f: Vec<&str> = line.split_whitespace().collect();
            if f.len() != 3 {
                return Err(format!("bad manifest line: {line}"));
            }
            let id: usize = f[0].parse().map_err(|_| format!("bad id in {line}"))?;
            let translevel: u8 = f[1].parse().map_err(|_| format!("bad translevel in {line}"))?;
            let state = std::fs::read(dir.join(format!("{id}.state"))).map_err(|e| e.to_string())?;
            levels.push(Level { id, translevel, held_out: f[2] == "held_out", state });
        }
        Ok(Self { levels })
    }
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test --lib levels`
Expected: 2 passed.

- [ ] **Step 5: Write `examples/make_levels.rs`**

```rust
//! Warps into every translevel, keeps the horizontal ones where a biased-random policy makes
//! progress, and writes them to LEVELS_DIR (default `levels`). Env: CORE, LEVELS_DIR.
use burn_mario_rl::emulator::Emulator;
use burn_mario_rl::env::{MarioEnv, Outcome, overworld_state};
use burn_mario_rl::levels::{Level, LevelSet, assign_split};
use burn_mario_rl::ram::{self, warp_to_level};
use std::path::PathBuf;

fn main() -> Result<(), String> {
    let core = PathBuf::from(std::env::var("CORE").unwrap_or("/usr/lib/libretro/snes9x_libretro.so".into()));
    let rom = PathBuf::from("Super Mario World (USA).sfc");
    let out = PathBuf::from(std::env::var("LEVELS_DIR").unwrap_or("levels".into()));
    let mut emu = Emulator::load(&core, &rom)?;
    let ow = overworld_state(&mut emu)?;

    let mut kept: Vec<(u8, Vec<u8>)> = Vec::new();
    for tl in 0u8..=0x24 {
        let state = match warp_to_level(&mut emu, &ow, tl) {
            Ok(s) => s,
            Err(e) => {
                println!("tl {tl:#04x}: skip ({e})");
                continue;
            }
        };
        if ram::is_vertical(emu.ram()) {
            println!("tl {tl:#04x}: skip (vertical)");
            continue;
        }
        // Progress probe: biased-random policy, best of 3 tries.
        let mut env = MarioEnv::new(&core, &rom, state.clone())?;
        let mut rng = 99u64 + tl as u64;
        let mut best = 0u16;
        for _ in 0..3 {
            env.reset()?;
            for _ in 0..600 {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let a = if rng >> 60 < 9 { 4 } else { (rng >> 33) as usize % burn_mario_rl::env::ACTIONS.len() };
                if env.step(a).outcome != Outcome::Running {
                    break;
                }
            }
            best = best.max(env.max_x());
        }
        if best < 300 {
            println!("tl {tl:#04x}: skip (random policy reached only x={best})");
            continue;
        }
        println!("tl {tl:#04x}: keep (random policy reached x={best})");
        kept.push((tl, state));
    }

    let split = assign_split(kept.len());
    let levels = kept.into_iter().zip(split).enumerate().map(|(id, ((translevel, state), held_out))| Level { id, translevel, held_out, state }).collect();
    let set = LevelSet { levels };
    set.save(&out)?;
    println!("wrote {} levels ({} held out) to {}", set.levels.len(), set.held_out_ids().len(), out.display());
    Ok(())
}
```

- [ ] **Step 6: Run it and review the result**

Run: `cargo run --release --example make_levels 2>&1 | tail -50`
Expected: a keep/skip line per translevel and a final `wrote N levels (M held out)`. Report N to the user. If N < 15 that is acceptable (see Global Constraints) but state it plainly. If N < 5, stop and ask the user, since there is no held-out split.

- [ ] **Step 7: Commit**

```bash
git add src/levels.rs src/lib.rs examples/make_levels.rs
git commit -m "Add LevelSet and make_levels example"
```

---

### Task 3: Speed-scaled reward (pure functions)

**Files:**
- Create: `src/reward.rs`
- Modify: `src/lib.rs` (`pub mod reward;`)

**Interfaces:**
- Consumes: `ram::TIMER_TICK_FRAMES`, `env::FRAME_SKIP`
- Produces:
  - `pub fn clear_bonus(timer_left: u32, timer_start: u32, k: f32) -> f32`
  - `pub fn step_cap(timer_start: u32) -> u32`

- [ ] **Step 1: Write the failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bonus_scales_with_time_left() {
        assert_eq!(clear_bonus(0, 300, 1.0), 100.0);
        assert_eq!(clear_bonus(300, 300, 1.0), 200.0);
        assert_eq!(clear_bonus(150, 300, 1.0), 150.0);
        assert_eq!(clear_bonus(150, 300, 0.0), 100.0);
    }

    #[test]
    fn bonus_handles_zero_start_timer() {
        assert_eq!(clear_bonus(0, 0, 1.0), 100.0);
    }

    #[test]
    fn cap_follows_timer() {
        // timer ticks every TIMER_TICK_FRAMES frames; one agent step is FRAME_SKIP frames.
        let cap = step_cap(300);
        assert_eq!(cap, 300 * crate::ram::TIMER_TICK_FRAMES / crate::env::FRAME_SKIP);
        assert!(step_cap(0) >= 600, "never below the floor");
    }
}
```

- [ ] **Step 2: Run to verify they fail**

Run: `cargo test --lib reward`
Expected: compile error (module missing).

- [ ] **Step 3: Implement**

```rust
//! Reward terms that do not need the emulator, so they can be tested directly.

use crate::env::FRAME_SKIP;
use crate::ram::TIMER_TICK_FRAMES;

pub const REWARD_CLEAR: f32 = 100.0;
const MIN_STEP_CAP: u32 = 600;

/// Bonus for clearing the level: `100 * (1 + k * fraction of the timer left)`.
pub fn clear_bonus(timer_left: u32, timer_start: u32, k: f32) -> f32 {
    let frac = if timer_start == 0 { 0.0 } else { (timer_left.min(timer_start) as f32) / timer_start as f32 };
    REWARD_CLEAR * (1.0 + k * frac)
}

/// Agent steps until the in-game timer would reach zero, with a floor for short timers.
pub fn step_cap(timer_start: u32) -> u32 {
    (timer_start * TIMER_TICK_FRAMES / FRAME_SKIP).max(MIN_STEP_CAP)
}
```

The test `cap_follows_timer` for `step_cap(300)` requires `300 * TIMER_TICK_FRAMES / 4 >= 600`; with the measured constant from Task 1 this holds (40 gives 3000). If the measured constant makes it lower, change the test expectation to `.max(600)` of the formula.

- [ ] **Step 4: Run tests**

Run: `cargo test --lib reward`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/reward.rs src/lib.rs
git commit -m "Add speed-scaled clear bonus and timer step cap"
```

---

### Task 4: Multi-level `MarioEnv`

**Files:**
- Modify: `src/env.rs`
- Modify: `examples/env_check.rs`, `examples/make_curriculum.rs`, `src/bin/eval.rs` only if they stop compiling

**Interfaces:**
- Consumes: `LevelSet`, `reward::{clear_bonus, step_cap}`, `ram::timer`
- Produces:
  - `MarioEnv::with_levels(core: &Path, rom: &Path, levels: Arc<LevelSet>, k_speed: f32) -> Result<Self, String>`
  - `MarioEnv::reset_to(&mut self, level: usize) -> Result<&[u8], String>`
  - `MarioEnv::level(&self) -> usize` (index into `LevelSet.levels` of the current episode)
  - `MarioEnv::new(...)` keeps its signature (single level, `k_speed = 0.0` is NOT used; it uses k = 1.0)
  - `StepResult` gains nothing; `Outcome` unchanged.

The existing `with_starts` / `p_start` / `start_index` curriculum code is removed: the multi-level set replaces it. `new` wraps the single state in a one-level `LevelSet`.

- [ ] **Step 1: Replace curriculum fields**

In `MarioEnv`, remove `starts`, `p_start`, `rng`, `current_start`, `next_random`, `start_index`, `with_starts`. Add:

```rust
    levels: Arc<LevelSet>,
    level: usize,
    k_speed: f32,
    timer_start: u32,
    step_cap: u32,
```

Constructors:

```rust
    pub fn new(core: &Path, rom: &Path, start_state: Vec<u8>) -> Result<Self, String> {
        let set = LevelSet { levels: vec![Level { id: 0, translevel: 0, held_out: false, state: start_state }] };
        Self::with_levels(core, rom, Arc::new(set), 1.0)
    }

    pub fn with_levels(core: &Path, rom: &Path, levels: Arc<LevelSet>, k_speed: f32) -> Result<Self, String> {
        let mut env = Self {
            emu: Emulator::load(core, rom)?,
            levels,
            level: 0,
            k_speed,
            timer_start: 0,
            step_cap: 0,
            obs: vec![0; OBS_LEN],
            prev_x: 0,
            max_x: 0,
            steps: 0,
            stall: 0,
        };
        env.reset_to(0)?;
        Ok(env)
    }

    pub fn level(&self) -> usize {
        self.level
    }
```

- [ ] **Step 2: `reset` and `reset_to`**

```rust
    /// Restart the current level.
    pub fn reset(&mut self) -> Result<&[u8], String> {
        self.reset_to(self.level)
    }

    /// Start a new episode on `level` (an index into the level set).
    pub fn reset_to(&mut self, level: usize) -> Result<&[u8], String> {
        self.level = level;
        self.emu.load_state(&self.levels.levels[level].state)?;
        self.emu.set_buttons(0);
        self.emu.run_frame(); // refresh the framebuffer after the state load
        self.prev_x = x_pos(&self.emu);
        self.max_x = self.prev_x;
        self.steps = 0;
        self.stall = 0;
        self.timer_start = crate::ram::timer(self.emu.ram());
        self.step_cap = crate::reward::step_cap(self.timer_start);
        let plane = OBS_W * OBS_H;
        grayscale_downscale(&self.emu, &mut self.obs[..plane]);
        for i in 1..STACK {
            self.obs.copy_within(..plane, i * plane);
        }
        Ok(&self.obs)
    }
```

- [ ] **Step 3: Reward wiring in `step_with`**

Remove the `REWARD_CLEAR` and `MAX_STEPS` constants from `env.rs` (they move to `reward.rs`). In `step_with` change the clear and timeout branches:

```rust
        let mut reward = dx * REWARD_PER_TILE - PENALTY_STEP;
        let outcome = if cleared {
            reward += crate::reward::clear_bonus(crate::ram::timer(self.emu.ram()), self.timer_start, self.k_speed);
            Outcome::Cleared
        } else if dying {
            reward -= PENALTY_DEATH;
            Outcome::Died
        } else if self.steps >= self.step_cap || self.stall >= MAX_STALL {
            Outcome::Timeout
        } else {
            Outcome::Running
        };
```

Add `use crate::levels::{Level, LevelSet}; use std::sync::Arc;`.

- [ ] **Step 4: Fix the dependents, build**

`env_check.rs` uses `MarioEnv::new` and `env.reset()`: unchanged. `make_curriculum.rs`/`eval.rs` use `MarioEnv::new`: unchanged. `train.rs` and `vec_env.rs` use `with_starts`; they are rewritten in Tasks 6 and 10, so for this commit make them compile by temporarily leaving them (they will fail). To keep the tree green, do Tasks 4 and 6 back to back and run `cargo check` after Task 6. Run now only: `cargo check --lib 2>&1 | head -30`
Expected: errors only in `vec_env.rs` (`with_starts` missing). Nothing in `env.rs`.

- [ ] **Step 5: Verify against the real level**

Run: `cargo run --release --example env_check 2>&1 | tail -25`
Expected: the final line prints agent steps per second (~400+), and some episodes `Cleared`. Cleared returns should now exceed 100 (bonus includes the timer part). If `env_check` does not compile because it needs `vec_env`, it does not (it imports only `env`); it should build.

- [ ] **Step 6: Commit (together with Task 6 if the tree is red)**

If `cargo check` is red because of `vec_env.rs`, do not commit yet; continue to Task 5 and Task 6, then commit Tasks 4+6 together with message `Multi-level env with shared level sampler`.

---

### Task 5: PLR sampler (pure logic, no ROM)

**Files:**
- Create: `src/rng.rs`, `src/plr.rs`
- Modify: `src/lib.rs` (`pub mod rng; pub mod plr;`)

**Interfaces:**
- Produces:
  - `rng::next_u64(state: &mut u64) -> u64`, `rng::unit_f32(state: &mut u64) -> f32` (uniform in [0,1))
  - `pub struct PlrConfig { pub enabled: bool, pub temperature: f32, pub rho: f32, pub ema: f32 }` with `Default`
  - `pub struct LevelSampler`
  - `LevelSampler::new(ids: Vec<usize>, cfg: PlrConfig) -> Self`
  - `LevelSampler::sample(&mut self, rng: &mut u64) -> usize` (returns a level id from `ids`)
  - `LevelSampler::update(&mut self, level: usize, score: f32)` (EMA)
  - `LevelSampler::weights(&self) -> Vec<(usize, f32)>` (sampling probabilities, sums to 1)
  - `LevelSampler::scores(&self) -> Vec<(usize, f32)>`
  - `pub fn level_scores(adv: &[f32], level_ids: &[usize], num_levels: usize) -> Vec<Option<f32>>`
  - `pub type SharedSampler = std::sync::Arc<std::sync::Mutex<LevelSampler>>`

- [ ] **Step 1: `src/rng.rs`**

```rust
//! Tiny xorshift generator shared by the sampler and the augmentation.

pub fn next_u64(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

/// Uniform in [0, 1).
pub fn unit_f32(state: &mut u64) -> f32 {
    (next_u64(state) >> 40) as f32 / (1u64 << 24) as f32
}
```

- [ ] **Step 2: Write the failing tests in `src/plr.rs`**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(enabled: bool) -> PlrConfig {
        PlrConfig { enabled, temperature: 0.3, rho: 0.1, ema: 0.5 }
    }

    #[test]
    fn disabled_is_uniform() {
        let s = LevelSampler::new(vec![3, 7, 9], cfg(false));
        for (_, w) in s.weights() {
            assert!((w - 1.0 / 3.0).abs() < 1e-6);
        }
    }

    #[test]
    fn weights_sum_to_one_and_favor_high_scores() {
        let mut s = LevelSampler::new(vec![0, 1, 2, 3], cfg(true));
        s.update(0, 0.1);
        s.update(1, 5.0);
        s.update(2, 1.0);
        s.update(3, 0.5);
        let w = s.weights();
        let total: f32 = w.iter().map(|(_, p)| p).sum();
        assert!((total - 1.0).abs() < 1e-5);
        let p = |id: usize| w.iter().find(|(i, _)| *i == id).unwrap().1;
        assert!(p(1) > p(2) && p(2) > p(3) && p(3) > p(0));
    }

    #[test]
    fn staleness_lifts_a_level_that_has_not_been_sampled() {
        let mut s = LevelSampler::new(vec![0, 1], PlrConfig { enabled: true, temperature: 0.3, rho: 0.9, ema: 0.5 });
        s.update(0, 1.0);
        s.update(1, 1.0);
        for _ in 0..50 {
            s.note_sampled(0);
        }
        let w = s.weights();
        let p = |id: usize| w.iter().find(|(i, _)| *i == id).unwrap().1;
        assert!(p(1) > p(0), "level 1 is staler");
    }

    #[test]
    fn tied_scores_get_equal_weight() {
        // Before any update every score is 0; list order must not decide the weights.
        let s = LevelSampler::new(vec![0, 1, 2, 3], cfg(true));
        let w = s.weights();
        for (_, p) in &w {
            assert!((p - 0.25).abs() < 1e-6, "got {w:?}");
        }
        let mut s = LevelSampler::new(vec![0, 1, 2], cfg(true));
        s.update(2, 3.0); // one clear leader, two tied behind it
        let w = s.weights();
        let p = |id: usize| w.iter().find(|(i, _)| *i == id).unwrap().1;
        assert!(p(2) > p(0));
        assert!((p(0) - p(1)).abs() < 1e-6);
    }

    #[test]
    fn sample_only_returns_known_ids_and_covers_all() {
        let mut s = LevelSampler::new(vec![4, 8, 15], cfg(true));
        let mut rng = 12345u64;
        let mut seen = std::collections::HashSet::new();
        for _ in 0..500 {
            let id = s.sample(&mut rng);
            assert!([4, 8, 15].contains(&id));
            seen.insert(id);
        }
        assert_eq!(seen.len(), 3);
    }

    #[test]
    fn ema_blends_old_and_new() {
        let mut s = LevelSampler::new(vec![0], cfg(true));
        s.update(0, 2.0);
        assert_eq!(s.scores()[0].1, 2.0, "first update sets the score");
        s.update(0, 4.0);
        assert_eq!(s.scores()[0].1, 3.0, "ema 0.5 of 2 and 4");
    }

    #[test]
    fn level_scores_average_abs_advantage_per_level() {
        let adv = [1.0, -3.0, 2.0, 0.5];
        let ids = [0, 0, 2, 2];
        let s = level_scores(&adv, &ids, 3);
        assert_eq!(s[0], Some(2.0));
        assert_eq!(s[1], None);
        assert_eq!(s[2], Some(1.25));
    }
}
```

- [ ] **Step 3: Run to verify they fail**

Run: `cargo test --lib plr`
Expected: compile error (module missing).

- [ ] **Step 4: Implement `src/plr.rs`**

```rust
//! Prioritized Level Replay over a fixed set of levels. A level's score is the mean |GAE| seen
//! on it (how surprised the value function still is). Sampling mixes a rank-based score weight
//! with a staleness weight, so no level is starved.

use crate::rng::unit_f32;
use std::sync::{Arc, Mutex};

#[derive(Clone, Copy, Debug)]
pub struct PlrConfig {
    /// false = uniform over the levels.
    pub enabled: bool,
    /// Lower = sharper preference for high-score levels.
    pub temperature: f32,
    /// Share of the staleness distribution in the mix.
    pub rho: f32,
    /// EMA rate for score updates.
    pub ema: f32,
}

impl Default for PlrConfig {
    fn default() -> Self {
        Self { enabled: true, temperature: 0.3, rho: 0.1, ema: 0.3 }
    }
}

pub struct LevelSampler {
    cfg: PlrConfig,
    ids: Vec<usize>,
    scores: Vec<f32>,
    seen: Vec<bool>,
    /// Value of `clock` when each level was last sampled.
    last_sampled: Vec<u64>,
    clock: u64,
}

pub type SharedSampler = Arc<Mutex<LevelSampler>>;

impl LevelSampler {
    pub fn new(ids: Vec<usize>, cfg: PlrConfig) -> Self {
        let n = ids.len();
        Self { cfg, ids, scores: vec![0.0; n], seen: vec![false; n], last_sampled: vec![0; n], clock: 0 }
    }

    pub fn shared(ids: Vec<usize>, cfg: PlrConfig) -> SharedSampler {
        Arc::new(Mutex::new(Self::new(ids, cfg)))
    }

    pub fn update(&mut self, level: usize, score: f32) {
        let Some(i) = self.ids.iter().position(|&id| id == level) else { return };
        self.scores[i] = if self.seen[i] { (1.0 - self.cfg.ema) * self.scores[i] + self.cfg.ema * score } else { score };
        self.seen[i] = true;
    }

    /// Record that `level` was just handed out.
    pub fn note_sampled(&mut self, level: usize) {
        if let Some(i) = self.ids.iter().position(|&id| id == level) {
            self.clock += 1;
            self.last_sampled[i] = self.clock;
        }
    }

    pub fn scores(&self) -> Vec<(usize, f32)> {
        self.ids.iter().copied().zip(self.scores.iter().copied()).collect()
    }

    /// Probability of handing out each level.
    pub fn weights(&self) -> Vec<(usize, f32)> {
        let n = self.ids.len();
        if !self.cfg.enabled {
            return self.ids.iter().map(|&id| (id, 1.0 / n as f32)).collect();
        }
        // Rank 1 = highest score; weight (1 / rank)^(1 / temperature). Tied scores share the
        // average weight of the ranks they span, so list order never breaks a tie.
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| self.scores[b].total_cmp(&self.scores[a]));
        let rank_weight = |rank: usize| (1.0 / (rank as f32 + 1.0)).powf(1.0 / self.cfg.temperature);
        let mut p_score = vec![0.0f32; n];
        let mut start = 0;
        while start < n {
            let mut end = start + 1;
            while end < n && self.scores[order[end]] == self.scores[order[start]] {
                end += 1;
            }
            let shared = (start..end).map(rank_weight).sum::<f32>() / (end - start) as f32;
            for &i in &order[start..end] {
                p_score[i] = shared;
            }
            start = end;
        }
        let total: f32 = p_score.iter().sum();
        p_score.iter_mut().for_each(|p| *p /= total);
        // Staleness: how long ago each level was last handed out.
        let mut p_stale: Vec<f32> = self.last_sampled.iter().map(|&t| (self.clock - t) as f32 + 1.0).collect();
        let total: f32 = p_stale.iter().sum();
        p_stale.iter_mut().for_each(|p| *p /= total);
        (0..n).map(|i| (self.ids[i], (1.0 - self.cfg.rho) * p_score[i] + self.cfg.rho * p_stale[i])).collect()
    }

    pub fn sample(&mut self, rng: &mut u64) -> usize {
        let weights = self.weights();
        let mut u = unit_f32(rng);
        let mut chosen = weights[weights.len() - 1].0;
        for &(id, p) in &weights {
            if u < p {
                chosen = id;
                break;
            }
            u -= p;
        }
        self.note_sampled(chosen);
        chosen
    }
}

/// Mean |advantage| per level over one rollout. `None` for levels that did not appear.
pub fn level_scores(adv: &[f32], level_ids: &[usize], num_levels: usize) -> Vec<Option<f32>> {
    let (mut sum, mut count) = (vec![0.0f32; num_levels], vec![0u32; num_levels]);
    for (a, &l) in adv.iter().zip(level_ids) {
        sum[l] += a.abs();
        count[l] += 1;
    }
    (0..num_levels).map(|l| (count[l] > 0).then(|| sum[l] / count[l] as f32)).collect()
}
```

- [ ] **Step 5: Run tests**

Run: `cargo test --lib plr`
Expected: 7 passed. If `staleness_lifts...` fails, check that `note_sampled` increments `clock` before storing (stale weight for level 0 must drop).

- [ ] **Step 6: Commit**

```bash
git add src/rng.rs src/plr.rs src/lib.rs
git commit -m "Add PLR level sampler"
```

---

### Task 6: `VecEnv` with shared sampler and level ids

**Files:**
- Modify: `src/vec_env.rs`

**Interfaces:**
- Consumes: `MarioEnv::{with_levels, reset_to, level}`, `SharedSampler`, `Arc<LevelSet>`
- Produces:
  - `VecEnv::new(n: usize, core: &Path, rom: &Path, levels: Arc<LevelSet>, sampler: SharedSampler, k_speed: f32, seed: u64) -> Result<Self, String>`
  - `VecEnv.levels: Vec<usize>` field (level of the episode each env is currently in, updated every step)
  - `StepBatch.step_levels: Vec<usize>` (the level each env was in when it took the step; differs from `VecEnv.levels` only on episode boundaries)
  - `EpisodeInfo.level: usize` replaces `start`

- [ ] **Step 1: Update types**

Replace `start: usize` in `EpisodeInfo` with `level: usize`. Add `level: usize` and `step_level: usize` to `Reply`. Add `step_levels: Vec<usize>` to `StepBatch`, and `pub levels: Vec<usize>` to `VecEnv`.

- [ ] **Step 2: Worker**

```rust
fn worker(id: usize, core: std::path::PathBuf, rom: std::path::PathBuf, levels: Arc<LevelSet>, sampler: SharedSampler, k_speed: f32, seed: u64, commands: Receiver<usize>, replies: Sender<Reply>) {
    let mut rng = seed.wrapping_mul(0x9E3779B97F4A7C15) | 1;
    let mut env = match MarioEnv::with_levels(&core, &rom, levels, k_speed) {
        Ok(env) => env,
        Err(e) => {
            eprintln!("worker {id}: {e}");
            return;
        }
    };
    let _ = std::fs::remove_file(&core); // the library stays mapped after unlinking
    let first = sampler.lock().expect("sampler").sample(&mut rng);
    let _ = env.reset_to(first);
    let _ = replies.send(Reply { id, obs: env.observation().to_vec(), reward: 0.0, done: false, info: None, level: env.level(), step_level: env.level() });
    let (mut ret, mut steps) = (0.0f32, 0u32);
    while let Ok(action) = commands.recv() {
        let step_level = env.level();
        let StepResult { reward, outcome } = env.step(action);
        ret += reward;
        steps += 1;
        let done = outcome != Outcome::Running;
        let info = done.then(|| EpisodeInfo { ret, steps, outcome, max_x: env.max_x(), level: step_level });
        if done {
            let next = sampler.lock().expect("sampler").sample(&mut rng);
            let _ = env.reset_to(next);
            ret = 0.0;
            steps = 0;
        }
        if replies.send(Reply { id, obs: env.observation().to_vec(), reward, done, info, level: env.level(), step_level }).is_err() {
            break;
        }
    }
}
```

- [ ] **Step 3: `VecEnv::new` and `step`**

`new` signature as in Interfaces; spawn each worker with `levels.clone()`, `sampler.clone()` and `seed + id as u64 + 1`. After the initial replies, fill `self.levels[r.id] = r.level`. In `step`, set `batch.step_levels[r.id] = r.step_level` and `self.levels[r.id] = r.level`. Add imports `use crate::levels::LevelSet; use crate::plr::SharedSampler; use std::sync::Arc;`.

- [ ] **Step 4: Build**

Run: `cargo check --lib`
Expected: clean. (`train.rs` still fails; fixed in Task 10.)

- [ ] **Step 5: Commit Tasks 4 and 6 together**

```bash
git add src/env.rs src/vec_env.rs
git commit -m "Multi-level env with shared level sampler"
```

---

### Task 7: Entity vector

**Files:**
- Create: `src/entities.rs`
- Modify: `src/lib.rs` (`pub mod entities;`), `src/env.rs`, `src/vec_env.rs`

**Interfaces:**
- Consumes: `ram` constants
- Produces:
  - `pub const ENT_LEN: usize = 66` (12 sprite slots x 5 + 6 Mario features)
  - `pub fn entity_vector(ram: &[u8], out: &mut [f32])` (writes exactly `ENT_LEN` values)
  - `MarioEnv::entities(&self) -> &[f32]`
  - `VecEnv.ent: Vec<f32>` (`n * ENT_LEN`)

Layout per sprite slot: `[present, type/255, dx/128, dy/128, status/15]`, `dx`/`dy` relative to Mario clamped to +-128 pixels. Mario block: `[speed_x/64, speed_y/64, on_ground, powerup/3, (x - camera_x)/256, y_on_screen/256]` where x - camera uses `$1A` (camera X, 16 bit) and y uses `$1C`.

- [ ] **Step 1: Failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::ram::*;

    fn blank() -> Vec<u8> {
        vec![0u8; 0x20000]
    }

    #[test]
    fn empty_slots_are_zero() {
        let ram = blank();
        let mut out = vec![9.0; ENT_LEN];
        entity_vector(&ram, &mut out);
        assert!(out[..SPRITE_SLOTS * 5].iter().all(|&v| v == 0.0));
    }

    #[test]
    fn sprite_is_encoded_relative_to_mario() {
        let mut ram = blank();
        ram[PLAYER_X..PLAYER_X + 2].copy_from_slice(&100u16.to_le_bytes());
        ram[PLAYER_Y..PLAYER_Y + 2].copy_from_slice(&200u16.to_le_bytes());
        ram[SPRITE_STATUS + 2] = 8;
        ram[SPRITE_TYPE + 2] = 51;
        ram[SPRITE_X_LO + 2] = 164; // 64 pixels right of Mario
        ram[SPRITE_Y_LO + 2] = 180; // 20 pixels above
        let mut out = vec![0.0; ENT_LEN];
        entity_vector(&ram, &mut out);
        let s = &out[2 * 5..3 * 5];
        assert_eq!(s[0], 1.0);
        assert!((s[1] - 51.0 / 255.0).abs() < 1e-6);
        assert!((s[2] - 0.5).abs() < 1e-6);
        assert!((s[3] + 20.0 / 128.0).abs() < 1e-6);
        assert!((s[4] - 8.0 / 15.0).abs() < 1e-6);
    }

    #[test]
    fn far_sprites_are_clamped() {
        let mut ram = blank();
        ram[SPRITE_STATUS] = 8;
        ram[SPRITE_X_LO] = 0;
        ram[SPRITE_X_HI] = 5; // x = 1280
        let mut out = vec![0.0; ENT_LEN];
        entity_vector(&ram, &mut out);
        assert_eq!(out[2], 1.0);
    }

    #[test]
    fn mario_speed_is_signed() {
        let mut ram = blank();
        ram[PLAYER_SPEED_X] = (-32i8) as u8;
        ram[PLAYER_BLOCKED] = 4;
        let mut out = vec![0.0; ENT_LEN];
        entity_vector(&ram, &mut out);
        let m = &out[SPRITE_SLOTS * 5..];
        assert_eq!(m[0], -0.5);
        assert_eq!(m[2], 1.0);
    }
}
```

- [ ] **Step 2: Run to verify they fail**

Run: `cargo test --lib entities`
Expected: compile error.

- [ ] **Step 3: Implement**

```rust
//! Entity observation: the 12 sprite slots plus Mario's motion, read straight from RAM.

use crate::ram::*;

pub const ENT_LEN: usize = SPRITE_SLOTS * 5 + 6;
const CAMERA_X: usize = 0x1A;
const CAMERA_Y: usize = 0x1C;

fn clamp_unit(v: i32, scale: f32) -> f32 {
    (v as f32 / scale).clamp(-1.0, 1.0)
}

pub fn entity_vector(ram: &[u8], out: &mut [f32]) {
    debug_assert_eq!(out.len(), ENT_LEN);
    let (mx, my) = (u16_at(ram, PLAYER_X) as i32, u16_at(ram, PLAYER_Y) as i32);
    for i in 0..SPRITE_SLOTS {
        let slot = &mut out[i * 5..(i + 1) * 5];
        let status = ram[SPRITE_STATUS + i];
        if status == 0 {
            slot.fill(0.0);
            continue;
        }
        let sx = ram[SPRITE_X_LO + i] as i32 | (ram[SPRITE_X_HI + i] as i32) << 8;
        let sy = ram[SPRITE_Y_LO + i] as i32 | (ram[SPRITE_Y_HI + i] as i32) << 8;
        slot[0] = 1.0;
        slot[1] = ram[SPRITE_TYPE + i] as f32 / 255.0;
        slot[2] = clamp_unit(sx - mx, 128.0);
        slot[3] = clamp_unit(sy - my, 128.0);
        slot[4] = status as f32 / 15.0;
    }
    let m = &mut out[SPRITE_SLOTS * 5..];
    m[0] = clamp_unit(ram[PLAYER_SPEED_X] as i8 as i32, 64.0);
    m[1] = clamp_unit(ram[PLAYER_SPEED_Y] as i8 as i32, 64.0);
    m[2] = (ram[PLAYER_BLOCKED] & 4 != 0) as u8 as f32;
    m[3] = ram[POWERUP] as f32 / 3.0;
    m[4] = clamp_unit(mx - u16_at(ram, CAMERA_X) as i32, 256.0);
    m[5] = clamp_unit(my - u16_at(ram, CAMERA_Y) as i32, 256.0);
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test --lib entities`
Expected: 4 passed.

- [ ] **Step 5: Wire into the env and VecEnv**

In `MarioEnv`: add field `ent: Vec<f32>` (init `vec![0.0; ENT_LEN]`), refresh it at the end of `reset_to` and `step_with` with `entity_vector(self.emu.ram(), &mut self.ent)` (take the ram slice into a local first to avoid a borrow conflict: `let ram = self.emu.ram(); entity_vector(ram, &mut self.ent);` works because `emu` and `ent` are distinct fields), and add:

```rust
    pub fn entities(&self) -> &[f32] {
        &self.ent
    }
```

In `vec_env.rs`: add `ent: Vec<f32>` to `Reply` (`env.entities().to_vec()`), `pub ent: Vec<f32>` to `VecEnv` (`vec![0.0; n * ENT_LEN]`), copy it in `new` and `step` the same way as `obs`.

- [ ] **Step 6: Check and commit**

Run: `cargo check --lib && cargo test --lib`
Expected: clean, all tests pass.

```bash
git add src/entities.rs src/lib.rs src/env.rs src/vec_env.rs
git commit -m "Add entity vector observation"
```

---

### Task 8: Network - `NetConfig`, IMPALA trunk, entity branch

**Files:**
- Modify: `src/ppo.rs`

**Interfaces:**
- Consumes: `entities::ENT_LEN`
- Produces:
  - `pub enum TrunkKind { Nature, Impala }`
  - `pub struct NetConfig { pub trunk: TrunkKind, pub width: usize, pub entities: bool }` with `Default` (Impala, width 1, entities true), `NetConfig::from_env()`, `NetConfig::to_line(&self) -> String`, `NetConfig::from_line(&str) -> Option<Self>`, `NetConfig::for_checkpoint(path: &str) -> Self`
  - `ActorCritic::<B>::new(cfg: &NetConfig, device: &B::Device) -> Self`
  - `ActorCritic::forward(&self, obs: Tensor<B, 4>, ent: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 1>)`
  - `pub fn ent_tensor<B: Backend>(floats: &[f32], batch: usize, device: &B::Device) -> Tensor<B, 2>`

`width` multiplies the IMPALA channel counts (16/32/32 x width). The Nature trunk ignores `width`.

- [ ] **Step 1: Failing tests (bottom of `ppo.rs`)**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    type T = NdArray;

    fn run(cfg: NetConfig) -> ([usize; 2], [usize; 1]) {
        let device = Default::default();
        let model = ActorCritic::<T>::new(&cfg, &device);
        let obs = Tensor::<T, 4>::zeros([3, STACK, OBS_H, OBS_W], &device);
        let ent = Tensor::<T, 2>::zeros([3, ENT_LEN], &device);
        let (logits, value) = model.forward(obs, ent);
        (logits.dims(), value.dims())
    }

    #[test]
    fn impala_shapes() {
        let (l, v) = run(NetConfig { trunk: TrunkKind::Impala, width: 1, entities: true });
        assert_eq!(l, [3, NUM_ACTIONS]);
        assert_eq!(v, [3]);
    }

    #[test]
    fn impala_wide_without_entities() {
        let (l, v) = run(NetConfig { trunk: TrunkKind::Impala, width: 2, entities: false });
        assert_eq!(l, [3, NUM_ACTIONS]);
        assert_eq!(v, [3]);
    }

    #[test]
    fn nature_shapes() {
        let (l, v) = run(NetConfig { trunk: TrunkKind::Nature, width: 1, entities: true });
        assert_eq!(l, [3, NUM_ACTIONS]);
        assert_eq!(v, [3]);
    }

    #[test]
    fn config_line_round_trip() {
        let cfg = NetConfig { trunk: TrunkKind::Impala, width: 2, entities: false };
        let back = NetConfig::from_line(&cfg.to_line()).unwrap();
        assert_eq!((back.trunk, back.width, back.entities), (TrunkKind::Impala, 2, false));
    }
}
```

`TrunkKind` must derive `Clone, Copy, Debug, PartialEq`.

- [ ] **Step 2: Run to verify they fail**

Run: `cargo test --lib ppo`
Expected: compile errors.

- [ ] **Step 3: Implement the config**

```rust
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TrunkKind {
    Nature,
    Impala,
}

#[derive(Clone, Copy, Debug)]
pub struct NetConfig {
    pub trunk: TrunkKind,
    /// Channel multiplier for the IMPALA trunk (1 = 16/32/32).
    pub width: usize,
    pub entities: bool,
}

impl Default for NetConfig {
    fn default() -> Self {
        Self { trunk: TrunkKind::Impala, width: 1, entities: true }
    }
}

impl NetConfig {
    /// Env vars TRUNK (nature|impala), WIDTH, ENTITIES (0|1).
    pub fn from_env() -> Self {
        let d = Self::default();
        let trunk = match std::env::var("TRUNK").as_deref() {
            Ok("nature") => TrunkKind::Nature,
            Ok("impala") => TrunkKind::Impala,
            _ => d.trunk,
        };
        let width = std::env::var("WIDTH").ok().and_then(|v| v.parse().ok()).unwrap_or(d.width);
        let entities = std::env::var("ENTITIES").ok().map(|v| v != "0").unwrap_or(d.entities);
        Self { trunk, width, entities }
    }

    pub fn to_line(&self) -> String {
        format!("{} {} {}", if self.trunk == TrunkKind::Nature { "nature" } else { "impala" }, self.width, self.entities as u8)
    }

    pub fn from_line(line: &str) -> Option<Self> {
        let f: Vec<&str> = line.split_whitespace().collect();
        let trunk = match *f.first()? {
            "nature" => TrunkKind::Nature,
            "impala" => TrunkKind::Impala,
            _ => return None,
        };
        Some(Self { trunk, width: f.get(1)?.parse().ok()?, entities: *f.get(2)? != "0" })
    }

    /// Reads `net.cfg` next to a checkpoint (written by train); falls back to `from_env`.
    pub fn for_checkpoint(path: &str) -> Self {
        let dir = std::path::Path::new(path).parent().unwrap_or(std::path::Path::new("."));
        std::fs::read_to_string(dir.join("net.cfg")).ok().and_then(|s| Self::from_line(&s)).unwrap_or_else(Self::from_env)
    }
}
```

- [ ] **Step 4: Implement the trunks and model**

Imports to add: `burn::nn::PaddingConfig2d`, `burn::nn::pool::{MaxPool2d, MaxPool2dConfig}`, `crate::entities::ENT_LEN`, `burn::tensor::Tensor`.

```rust
#[derive(Module, Debug)]
pub struct NatureTrunk<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    conv3: Conv2d<B>,
    fc: Linear<B>,
}

impl<B: Backend> NatureTrunk<B> {
    fn new(device: &B::Device) -> Self {
        Self {
            conv1: Conv2dConfig::new([STACK, 32], [8, 8]).with_stride([4, 4]).init(device),
            conv2: Conv2dConfig::new([32, 64], [4, 4]).with_stride([2, 2]).init(device),
            conv3: Conv2dConfig::new([64, 64], [3, 3]).init(device),
            fc: LinearConfig::new(64 * 7 * 7, 512).init(device),
        }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        let x = relu(self.conv1.forward(x));
        let x = relu(self.conv2.forward(x));
        let x = relu(self.conv3.forward(x));
        let batch = x.dims()[0];
        relu(self.fc.forward(x.reshape([batch, 64 * 7 * 7])))
    }
}

#[derive(Module, Debug)]
pub struct ResBlock<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
}

impl<B: Backend> ResBlock<B> {
    fn new(ch: usize, device: &B::Device) -> Self {
        let conv = || Conv2dConfig::new([ch, ch], [3, 3]).with_padding(PaddingConfig2d::Same).init(device);
        Self { conv1: conv(), conv2: conv() }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let y = self.conv1.forward(relu(x.clone()));
        let y = self.conv2.forward(relu(y));
        x + y
    }
}

#[derive(Module, Debug)]
pub struct ImpalaStage<B: Backend> {
    conv: Conv2d<B>,
    pool: MaxPool2d,
    res1: ResBlock<B>,
    res2: ResBlock<B>,
}

impl<B: Backend> ImpalaStage<B> {
    fn new(c_in: usize, c_out: usize, device: &B::Device) -> Self {
        Self {
            conv: Conv2dConfig::new([c_in, c_out], [3, 3]).with_padding(PaddingConfig2d::Same).init(device),
            pool: MaxPool2dConfig::new([3, 3]).with_strides([2, 2]).with_padding(PaddingConfig2d::Explicit(1, 1)).init(),
            res1: ResBlock::new(c_out, device),
            res2: ResBlock::new(c_out, device),
        }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.pool.forward(self.conv.forward(x));
        self.res2.forward(self.res1.forward(x))
    }
}

/// IMPALA-CNN: three conv/pool/residual stages. 84x84 input becomes 11x11 feature maps.
#[derive(Module, Debug)]
pub struct ImpalaTrunk<B: Backend> {
    stages: Vec<ImpalaStage<B>>,
    fc: Linear<B>,
}

impl<B: Backend> ImpalaTrunk<B> {
    fn new(width: usize, device: &B::Device) -> Self {
        let ch = [16 * width, 32 * width, 32 * width];
        let stages = vec![ImpalaStage::new(STACK, ch[0], device), ImpalaStage::new(ch[0], ch[1], device), ImpalaStage::new(ch[1], ch[2], device)];
        Self { stages, fc: LinearConfig::new(ch[2] * 11 * 11, 256).init(device) }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        let mut x = x;
        for s in &self.stages {
            x = s.forward(x);
        }
        let [batch, c, h, w] = x.dims();
        relu(self.fc.forward(relu(x).reshape([batch, c * h * w])))
    }
}

#[derive(Module, Debug)]
pub enum Trunk<B: Backend> {
    Nature(NatureTrunk<B>),
    Impala(ImpalaTrunk<B>),
}

const ENT_HIDDEN: usize = 64;

#[derive(Module, Debug)]
pub struct ActorCritic<B: Backend> {
    trunk: Trunk<B>,
    ent_fc: Option<Linear<B>>,
    actor: Linear<B>,
    critic: Linear<B>,
}

impl<B: Backend> ActorCritic<B> {
    pub fn new(cfg: &NetConfig, device: &B::Device) -> Self {
        let (trunk, feat) = match cfg.trunk {
            TrunkKind::Nature => (Trunk::Nature(NatureTrunk::new(device)), 512),
            TrunkKind::Impala => (Trunk::Impala(ImpalaTrunk::new(cfg.width, device)), 256),
        };
        let (ent_fc, fused) = if cfg.entities {
            (Some(LinearConfig::new(ENT_LEN, ENT_HIDDEN).init(device)), feat + ENT_HIDDEN)
        } else {
            (None, feat)
        };
        Self { trunk, ent_fc, actor: LinearConfig::new(fused, NUM_ACTIONS).init(device), critic: LinearConfig::new(fused, 1).init(device) }
    }

    /// `obs` holds raw bytes `[batch, STACK, OBS_H, OBS_W]` scaled to 0..1 here; `ent` is `[batch, ENT_LEN]`.
    pub fn forward(&self, obs: Tensor<B, 4>, ent: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 1>) {
        let x = obs / 255.0;
        let mut feat = match &self.trunk {
            Trunk::Nature(t) => t.forward(x),
            Trunk::Impala(t) => t.forward(x),
        };
        if let Some(fc) = &self.ent_fc {
            feat = Tensor::cat(vec![feat, relu(fc.forward(ent))], 1);
        }
        (self.actor.forward(feat.clone()), self.critic.forward(feat).squeeze_dim::<1>(1))
    }
}

pub fn ent_tensor<B: Backend>(floats: &[f32], batch: usize, device: &B::Device) -> Tensor<B, 2> {
    debug_assert_eq!(floats.len(), batch * ENT_LEN);
    Tensor::from_data(TensorData::new(floats.to_vec(), [batch, ENT_LEN]), device)
}
```

Delete the old `ActorCritic` struct and its `impl`. If `#[derive(Module)]` rejects the `Trunk` enum or `Vec<ImpalaStage>`, the fallback is: make `Trunk` a struct holding `nature: Option<NatureTrunk<B>>` and `impala: Option<ImpalaTrunk<B>>` with the same `forward` dispatch. Use the fallback only if the compile error says so.

- [ ] **Step 5: Update `ppo_step` and callers of `forward`**

`Batch` gains `pub ent: &'a [f32]`. In `ppo_step` build `let ent = ent_tensor::<B>(batch.ent, n, device);` and call `model.forward(obs, ent)`. (The DrAC term is added in Task 9.) Fix `bench_net.rs` to build `ActorCritic::<B>::new(&NetConfig::from_env(), &device)` and pass a zeros entity tensor `Tensor::<B, 2>::zeros([batch, ENT_LEN], &device)` to `forward`.

- [ ] **Step 6: Run tests**

Run: `cargo test --lib ppo`
Expected: 4 passed (shape tests run a full forward pass on ndarray; the first run is slow, that is fine).

- [ ] **Step 7: Benchmark the trunk (local, informational)**

Run: `TRUNK=impala WIDTH=1 cargo run --release --example bench_net` and again with `TRUNK=nature` and with `WIDTH=2`.
Expected: a table of ms per batch. Record the numbers in the commit message; no pass/fail. The T4 numbers get measured in Task 11.

- [ ] **Step 8: Commit**

```bash
git add src/ppo.rs examples/bench_net.rs
git commit -m "Add IMPALA trunk, entity branch and NetConfig"
```

---

### Task 9: DrAC augmentation

**Files:**
- Create: `src/augment.rs`
- Modify: `src/lib.rs` (`pub mod augment;`), `src/ppo.rs`

**Interfaces:**
- Consumes: `rng::next_u64`
- Produces:
  - `augment::random_shift(src: &[u8], out: &mut [u8], samples: usize, pad: usize, rng: &mut u64)` (per sample, one shift in `[-pad, pad]` for both axes, applied to all stack planes, edge-replicated)
  - `Hyper.aug_coef: f32`
  - `Batch.aug_obs: Option<&'a [u8]>`
  - `UpdateStats.aug_loss: f32`

- [ ] **Step 1: Failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::env::{OBS_H, OBS_LEN, OBS_W};

    fn ramp() -> Vec<u8> {
        (0..OBS_LEN).map(|i| (i % 251) as u8).collect()
    }

    #[test]
    fn zero_pad_is_identity() {
        let src = ramp();
        let mut out = vec![0u8; OBS_LEN];
        let mut rng = 1u64;
        random_shift(&src, &mut out, 1, 0, &mut rng);
        assert_eq!(src, out);
    }

    #[test]
    fn shift_plane_moves_pixels_and_replicates_edges() {
        let mut src = vec![0u8; OBS_H * OBS_W];
        for y in 0..OBS_H {
            for x in 0..OBS_W {
                src[y * OBS_W + x] = x as u8;
            }
        }
        let mut dst = vec![0u8; OBS_H * OBS_W];
        shift_plane(&src, &mut dst, 2, 0);
        assert_eq!(dst[10], 12, "pixel at x=10 now shows what was at x=12");
        assert_eq!(dst[OBS_W - 1], (OBS_W - 1) as u8, "right edge is replicated");
    }

    #[test]
    fn same_shift_for_every_plane_of_a_sample() {
        let src = ramp();
        let mut out = vec![0u8; OBS_LEN];
        let mut rng = 42u64;
        random_shift(&src, &mut out, 1, 4, &mut rng);
        let plane = OBS_H * OBS_W;
        // Planes of the ramp are offset copies of each other, so a shared shift keeps that offset.
        for p in 1..crate::env::STACK {
            let a = &out[..plane];
            let b = &out[p * plane..(p + 1) * plane];
            let (sa, sb) = (&src[..plane], &src[p * plane..(p + 1) * plane]);
            let differs_src = sa.iter().zip(sb).any(|(x, y)| x != y);
            let differs_out = a.iter().zip(b).any(|(x, y)| x != y);
            assert_eq!(differs_src, differs_out);
        }
    }
}
```

- [ ] **Step 2: Run to verify they fail**

Run: `cargo test --lib augment`
Expected: compile error.

- [ ] **Step 3: Implement**

```rust
//! Random-shift augmentation on observation bytes: pad by edge replication, crop back, which is
//! the same as sampling each output pixel from the clamped shifted coordinate.

use crate::env::{OBS_H, OBS_LEN, OBS_W, STACK};
use crate::rng::next_u64;

/// Output (x, y) shows source (x + dx, y + dy), clamped to the image.
pub fn shift_plane(src: &[u8], dst: &mut [u8], dx: i32, dy: i32) {
    for y in 0..OBS_H {
        let sy = (y as i32 + dy).clamp(0, OBS_H as i32 - 1) as usize;
        for x in 0..OBS_W {
            let sx = (x as i32 + dx).clamp(0, OBS_W as i32 - 1) as usize;
            dst[y * OBS_W + x] = src[sy * OBS_W + sx];
        }
    }
}

pub fn random_shift(src: &[u8], out: &mut [u8], samples: usize, pad: usize, rng: &mut u64) {
    let plane = OBS_H * OBS_W;
    let span = 2 * pad as u64 + 1;
    for s in 0..samples {
        let dx = (next_u64(rng) % span) as i32 - pad as i32;
        let dy = (next_u64(rng) % span) as i32 - pad as i32;
        for p in 0..STACK {
            let o = s * OBS_LEN + p * plane;
            shift_plane(&src[o..o + plane], &mut out[o..o + plane], dx, dy);
        }
    }
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test --lib augment`
Expected: 3 passed.

- [ ] **Step 5: DrAC loss in `ppo_step`**

Add `pub aug_coef: f32` to `Hyper`, `pub aug_obs: Option<&'a [u8]>` to `Batch`, `pub aug_loss: f32` to `UpdateStats`. After computing `entropy` and before `loss`:

```rust
    let aug_loss = match batch.aug_obs {
        Some(aug) if hyper.aug_coef > 0.0 => {
            let (aug_logits, aug_value) = model.forward(obs_tensor::<B>(aug, n, device), ent.clone());
            // Keep the policy and value on augmented frames equal to the clean ones.
            let clean_lp = log_softmax(logits.clone().detach(), 1);
            let kl = (clean_lp.clone().exp() * (clean_lp - log_softmax(aug_logits, 1))).sum_dim(1).mean();
            let value_gap = (aug_value - value.clone().detach()).powf_scalar(2.0).mean() * 0.5;
            Some(kl + value_gap)
        }
        _ => None,
    };
```

(`ent` must be cloned before the clean forward: change the clean call to `model.forward(obs, ent.clone())`.) Then:

```rust
    let mut loss = policy_loss.clone() + value_loss.clone() * hyper.value_coef - entropy.clone() * hyper.entropy_coef;
    let aug_stat = aug_loss.as_ref().map(|a| a.clone().into_scalar().elem::<f32>()).unwrap_or(0.0);
    if let Some(a) = aug_loss {
        loss = loss + a * hyper.aug_coef;
    }
```

and set `aug_loss: aug_stat` in `UpdateStats`.

- [ ] **Step 6: Test that the update runs and the term is zero for identical inputs**

Add to the `ppo` tests:

```rust
    #[test]
    fn ppo_step_runs_with_augmentation() {
        use burn::backend::Autodiff;
        use burn::optim::AdamConfig;
        type A = Autodiff<NdArray>;
        let device = Default::default();
        let cfg = NetConfig { trunk: TrunkKind::Nature, width: 1, entities: true };
        let model = ActorCritic::<A>::new(&cfg, &device);
        let mut optim = AdamConfig::new().init();
        let n = 4;
        let obs = vec![7u8; n * crate::env::OBS_LEN];
        let ent = vec![0.0f32; n * ENT_LEN];
        let batch = Batch {
            obs: &obs,
            ent: &ent,
            actions: &[0, 1, 2, 3],
            old_logp: &[-1.9; 4],
            advantages: &[0.1, -0.1, 0.2, -0.2],
            returns: &[0.0; 4],
            aug_obs: Some(&obs),
        };
        let hyper = Hyper { lr: 1e-4, clip: 0.2, value_coef: 0.5, entropy_coef: 0.01, epochs: 1, minibatches: 1, aug_coef: 0.1 };
        let (_, stats) = ppo_step(model, &mut optim, &hyper, &batch, &device);
        assert!(stats.policy_loss.is_finite() && stats.value_loss.is_finite());
        assert!(stats.aug_loss.abs() < 1e-5, "identical clean and augmented input gives zero consistency loss");
    }
```

Run: `cargo test --lib ppo`
Expected: 5 passed.

- [ ] **Step 7: Commit**

```bash
git add src/augment.rs src/lib.rs src/ppo.rs
git commit -m "Add random-shift augmentation with DrAC consistency loss"
```

---

### Task 10: Training loop integration

**Files:**
- Modify: `src/bin/train.rs`

**Interfaces:**
- Consumes: everything above.
- Env vars added: `LEVELS` (dir, default `levels`), `PLR` (0|1, default 1), `PLR_TEMP` (0.3), `PLR_RHO` (0.1), `PLR_EMA` (0.3), `SPEED_K` (1.0), `AUG_PAD` (4, 0 = off), `AUG_COEF` (0.1), `EVAL_EVERY` (20 iterations, 0 = off), `EVAL_ENVS` (4), `EVAL_STEPS` (600), plus `TRUNK`, `WIDTH`, `ENTITIES` from `NetConfig::from_env`. Removed: `STATE`, `CURRICULUM`, `START_PROB`, `INIT` stays (warm start from a same-architecture checkpoint).

- [ ] **Step 1: Setup**

Replace the state/curriculum block with:

```rust
    let levels = Arc::new(LevelSet::load(Path::new(&cfg("LEVELS", "levels".to_string())))?);
    let train_ids = levels.train_ids();
    let held_ids = levels.held_out_ids();
    println!("levels: {} train, {} held out", train_ids.len(), held_ids.len());
    let plr_cfg = PlrConfig { enabled: cfg::<u8>("PLR", 1) == 1, temperature: cfg("PLR_TEMP", 0.3), rho: cfg("PLR_RHO", 0.1), ema: cfg("PLR_EMA", 0.3) };
    let sampler = LevelSampler::shared(train_ids.clone(), plr_cfg);
    let k_speed: f32 = cfg("SPEED_K", 1.0);
    let net_cfg = NetConfig::from_env();
    let aug_pad: usize = cfg("AUG_PAD", 4);
    let aug_coef: f32 = if aug_pad > 0 { cfg("AUG_COEF", 0.1) } else { 0.0 };
```

Model: `ActorCritic::<B>::new(&net_cfg, &device)`. After `create_dir_all(&ckpt_dir)`: `std::fs::write(format!("{ckpt_dir}/net.cfg"), net_cfg.to_line())`. `Hyper` gets `aug_coef`. Create the env: `VecEnv::new(n, &core, &rom, levels.clone(), sampler.clone(), k_speed, seed)`.

- [ ] **Step 2: Buffers and rollout**

Add `ent_buf: Vec<f32>` (`batch_size * ENT_LEN`) and `level_buf: Vec<usize>` (`batch_size`). In the rollout loop, copy `envs.ent` into `ent_buf[t*n*ENT_LEN..]` before inference; call `inference.forward(obs_tensor(&envs.obs, n, &device), ent_tensor::<Inner>(&envs.ent, n, &device))` (same for `last_value`); after `envs.step`, `level_buf[range].copy_from_slice(&step.step_levels)`. Per-level stats: maintain `HashMap<usize, VecDeque<EpisodeInfo>>` of the last 50 finished episodes per level (replacing `recent_start`), keyed by `info.level`.

- [ ] **Step 3: PLR score update after GAE**

After `gae(...)`:

```rust
        let scores = level_scores(&adv, &level_buf, levels.levels.len());
        {
            let mut s = sampler.lock().expect("sampler");
            for (l, score) in scores.iter().enumerate() {
                if let Some(score) = score {
                    s.update(l, *score);
                }
            }
        }
```

(`update` ignores levels not in the sampler's ids, so held-out ids never enter training.)

- [ ] **Step 4: Minibatch assembly with augmentation**

Where each minibatch's `obs` is gathered, also gather `ent` (`chunk.iter().flat_map(...)`) and, when `aug_pad > 0`, build `aug` with `random_shift(&obs, &mut aug, chunk.len(), aug_pad, &mut rng)`. Pass `ent: &ent_mb` and `aug_obs: (aug_pad > 0).then_some(&aug[..])` in `Batch`. Replace the local xorshift in the shuffle with `burn_mario_rl::rng::next_u64(&mut rng)`.

- [ ] **Step 5: Held-out evaluation**

Before the loop, if `EVAL_EVERY > 0 && !held_ids.is_empty()`, create a second env: 

```rust
    let eval_sampler = LevelSampler::shared(held_ids.clone(), PlrConfig { enabled: false, ..plr_cfg });
    let mut eval_envs = VecEnv::new(eval_n, &core, &rom, levels.clone(), eval_sampler, k_speed, seed ^ 0xABCD)?;
```

Add a function in `train.rs`:

```rust
/// Run the held-out envs for `steps` agent steps with the current policy; return per-level (clears, episodes).
fn evaluate(model: &ActorCritic<Inner>, envs: &mut VecEnv, steps: usize, rng: &mut u64, device: &<Inner as Backend>::Device) -> BTreeMap<usize, (u32, u32, f32)> {
    let mut out: BTreeMap<usize, (u32, u32, f32)> = BTreeMap::new();
    for _ in 0..steps {
        let (logits, _) = model.forward(obs_tensor::<Inner>(&envs.obs, envs.n, device), ent_tensor::<Inner>(&envs.ent, envs.n, device));
        let logits = to_vec(logits);
        let chosen: Vec<usize> = (0..envs.n).map(|e| sample_action(&logits[e * NUM_ACTIONS..(e + 1) * NUM_ACTIONS], rng).0).collect();
        for info in envs.step(&chosen).finished {
            let e = out.entry(info.level).or_insert((0, 0, 0.0));
            e.1 += 1;
            if info.outcome == Outcome::Cleared {
                e.0 += 1;
                e.2 += info.steps as f32;
            }
        }
    }
    out
}
```

Every `EVAL_EVERY` iterations call `evaluate(&model.valid(), &mut eval_envs, eval_steps, &mut rng, &device)` and print a line per held-out level: `held-out level {id} (tl {translevel:#04x}): {clears}/{eps} cleared, mean clear time {t} steps`, plus a combined held-out clear rate. Add `use std::collections::{BTreeMap, HashMap}; use burn::tensor::backend::Backend;`.

Note for later reading: evaluation episodes are cut at `EVAL_STEPS`, so a level with a long clear time may show few episodes; raise `EVAL_STEPS` rather than judging from tiny counts.

- [ ] **Step 6: Logging and best checkpoint**

Replace the "from-start" figures with: train clear rate over all train levels (last 100 episodes), train return, mean clear time among clears, and a compact per-level line every 10 iterations: `L{id}:{clear%}`. Save `best` when the combined train clear rate (needs >= 20 episodes) exceeds the previous best, using the mean return as the tiebreak.

- [ ] **Step 7: Build and smoke test locally**

Run: `cargo build --release --bin train 2>&1 | tail -20`
Expected: builds.

Run (2 envs, tiny rollout, a few iterations; needs `levels/` from Task 2):
`ENVS=2 ROLLOUT=32 TOTAL_STEPS=640 EVAL_EVERY=5 EVAL_ENVS=2 EVAL_STEPS=64 CKPT_DIR=/tmp/smoke cargo run --release --bin train 2>&1 | tail -30`
Expected: prints `levels: N train, M held out`, 10 iterations with finite `pl vl ent`, one held-out evaluation block, `done:`, and `/tmp/smoke/net.cfg` exists. Run again with `PLR=0 AUG_PAD=0 TRUNK=nature ENTITIES=0` and confirm it also runs (the flags work).

- [ ] **Step 8: Commit**

```bash
git add src/bin/train.rs
git commit -m "Train on a level set with PLR, entities, augmentation and held-out evaluation"
```

---

### Task 11: Eval tool, remaining examples, docs, and the VM run

**Files:**
- Modify: `src/bin/eval.rs`, `examples/make_curriculum.rs`, `examples/env_check.rs` (only if it fails to build), `CLAUDE.md`

- [ ] **Step 1: `eval.rs` - level selection and entity input**

Env var `LEVEL` (level id from the manifest, default 0) with `LEVELS` dir. Load `LevelSet`, build `MarioEnv::with_levels(&core, &rom, levels, k)`, call `env.reset_to(level)` each episode. Load the model with `ActorCritic::<Inner>::new(&NetConfig::for_checkpoint(&ckpt), &device)`. Pass `ent_tensor::<Inner>(env.entities(), 1, &device)` to `forward`. Print `steps` as the clear time (already printed) and print mean clear steps at the end. Remove the `STATE` env var.

- [ ] **Step 2: `make_curriculum.rs`**

It exists to produce level-1 mid-level states for the old curriculum. Keep it compiling with minimal edits (new `forward` signature and `ActorCritic::new(&NetConfig::for_checkpoint(..), ..)`), or delete it with `git rm` if you cannot make it compile in a few minutes: the multi-level set replaces its purpose. Prefer deleting.

- [ ] **Step 3: Full check**

Run: `cargo test && cargo build --release --examples --bins 2>&1 | tail -20`
Expected: all tests pass; everything builds.

- [ ] **Step 4: Update `CLAUDE.md`**

Under "Goal and priorities" replace the "Out of scope" line with a note that other levels and generalization are now in scope (spec path). Under "Environment design" replace the curriculum paragraph with the level-set description (`levels/`, `make_levels`, PLR, speed bonus, entity vector, trunk flags), and under "Algorithm" mention DrAC and PLR. Add a "Findings" line recording the Task 1 results (working warp trigger, timer address and tick length, the number of levels kept).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "Eval by level, drop the single-level curriculum tool, update CLAUDE.md"
```

- [ ] **Step 6: STOP before any cloud work**

Report to the user: tests green, smoke run done, the number of levels, bench numbers. Ask for their OK before starting the stopped Huawei VM (`mario-rl-train`, see memory). Do not start it, upload anything, or delete anything without that yes. When they approve, the run plan is: start the VM, sync the repo and `levels/`, build with `--features cuda`, run `bench_net` once for IMPALA width 1 and 2 on the T4, then run three comparable configurations from scratch with the same `TOTAL_STEPS`: (a) baseline: `TRUNK=nature ENTITIES=0 PLR=0 AUG_PAD=0`; (b) full: defaults; (c) full with `PLR=0`. Compare held-out clear rate and clear time. Ask before stopping or deleting the VM when done.

---

## Self-Review

**Spec coverage**
- Spike: RAM warp, address verification → Task 1.
- Multi-level env, level per episode, shared sampler, `level_id` per step, horizontal filter → Tasks 2, 4, 6.
- Reward: speed-scaled bonus, timer-based cap, unchanged penalties → Tasks 3, 4.
- PLR with mean |GAE|, rank + staleness mix, off switch, ROM-free tests → Task 5; learner integration → Task 10.
- IMPALA trunk, width multiplier, selectable Nature, `bench_net` → Task 8.
- Entity vector (12 slots + Mario features, MLP fused) → Tasks 7, 8.
- Augmentation DrAC style, flag → Task 9, wired in Task 10.
- Evaluation: per-level and held-out clear rate, clear time, `eval` takes a level → Tasks 10, 11.
- Consequences: old checkpoints dropped (Global Constraints), VM only with OK, CLAUDE.md update → Task 11.

**Placeholder scan:** `TIMER_TICK_FRAMES` is written as 40 and replaced by the measured value in Task 1 Step 6 (a measurement, not a hole). Warp trigger candidates in Task 1 are an explicit search procedure with an acceptance table.

**Type consistency checked:** `NetConfig::new(&cfg, &device)` and `forward(obs, ent)` are used identically in Tasks 8, 9, 10, 11. `EpisodeInfo.level` (Task 6) is what Task 10 reads. `SharedSampler`, `LevelSampler::{shared, sample, update}` (Task 5) match their uses in Tasks 6 and 10. `Batch.ent` is added in Task 8 and `Batch.aug_obs` in Task 9; the Task 9 test constructs both.
