# burn-mario-rl

Reinforcement learning agent that beats the first level of Super Mario World (SNES). Written in Rust with [Burn](https://burn.dev). The end product is a **video (or live run) of the agent clearing the level**, to post on LinkedIn.

## Goal and priorities

1. **Milestone 1:** the agent clears the first level (the one after Yoshi's House; the map labels it Yoshi's Island 1). Nothing else matters until this works.
2. **Deliverable:** a clean MP4 of a winning run, recorded at native resolution, upscaled for posting.
3. **Speed of learning:** the user wants wall-clock training to be short. Pick the fastest GPU Colab offers and favor sample-efficient choices over elegant ones.

**Generalization is now in scope** (spec `docs/superpowers/specs/2026-10-04-generalization-design.md`, plan `docs/superpowers/plans/2026-10-04-generalization.md`): the agent trains on a pool of levels with held-out levels to measure transfer. Still out of scope: multi-game support.

## Stack and fallback

- **Primary:** Rust + Burn.
- **Fallback:** Python (stable-retro + PyTorch / Stable-Baselines3). Switch only when the Rust path is blocked, most likely by emulator bindings. Record the blocker in this file when switching.
- **Training runs on a Colab VM with a GPU**, headless. Use the `colab-operator` skill for sessions, file sync, and shell on the VM. Local machine is for development and smoke tests.
- **Colab test (T4, 2026-10-04):** T4 VM has only **2 vCPUs** and 12 GB RAM. CUDA 13 present; core via `apt install libretro-snes9x` (`/usr/lib/x86_64-linux-gnu/libretro/snes9x_libretro.so`); Rust via rustup. Env builds and gives results identical to local, at **180 agent steps/s per core** (2.4x slower than the dev machine). Examples read the core path from the `CORE` env var. `colab exec` times out on long jobs: start them with `Popen` in the kernel and poll a log file. `burn-cuda` not yet tested.
- **Compute plan:** emulation is CPU-bound (about 430 agent steps/s per core), so favor a big pay-per-use CPU VM (Huawei Cloud or Colab) with many parallel env processes, killed right after the run. The user is fine with a large VM for under an hour. A modest GPU handles PPO updates.
- **GPU:** the user has Colab Pro. An A100 may be overkill for a small CNN policy, since emulation on the CPU is often the bottleneck. Benchmark steps/sec on L4, T4 and A100 for one short run each, then pick the cheapest that keeps the GPU busy. Log the results here.
- Burn backend: `burn-cuda` or `wgpu` on Colab; `ndarray` locally for tests. Keep the backend a generic parameter so code is not tied to one.

## Emulator (biggest risk)

Rust has no mature SNES emulator crate, so the emulator integration is the first thing to de-risk.

- Preferred approach: load a **libretro core** (snes9x or bsnes) through a thin Rust libretro host. It gives frames, input, save states, and RAM access through one API.
- Scratch tools live in `examples/` (`explore` probes menus by RAM, `env_check` runs random episodes, `record_random` records a clear to video).
- Required capabilities: step N frames, set joypad input, read RAM, save and load state, grab the framebuffer.
- Build this as its own module behind an `Env` trait (`reset`, `step(action) -> (obs, reward, done)`) so the agent code does not know which emulator sits underneath.
- **Status:** `src/emulator.rs` is a working hand-written libretro host (`libloading`, no libretro crate: the crates.io ones are for writing cores). Proven with snes9x at `/usr/lib/libretro/snes9x_libretro.so` (pacman `libretro-snes9x`): frames, input, work RAM, save-state round trip.
- **Speed:** about 1800 fps per core on the dev machine (30x realtime), one emulator, no rendering to screen. Compare against Colab VM cores before choosing a platform.
- **Gotchas:** snes9x refuses to load unless the host accepts its requested pixel format (RGB565), so the host accepts all three formats. The ROM shows black for about 5 s (300 frames) at boot. The core keeps global state, so parallel envs need one process each, or a uniquely named copy of the `.so` per env.
- The ROM is supplied by the user, kept out of git (ROM extensions in `.gitignore`; it sits in the repo root), and is never committed or uploaded.

## Environment design

- Start from a **save state at level start**; `reset` loads it. This also removes the need for lives to matter, since a death returns to the level start anyway.
- **Status:** `src/env.rs` implements this and is verified. `level_start_state` plays the menus from power-on by watching game mode (`$7E0100`), and the state is cached in `level1.state` (gitignored; regenerate with `cargo run --release --example env_check`). Reward weights and caps are constants at the top of `env.rs`.
- **Verified clear detection:** `$1493 != 0` (or leaving level mode without dying) fires at the goal gate, and the game then shows COURSE CLEAR. Death is player state `$71 == 9`.
- **Level set (generalization):** `levels/` (gitignored, built by `cargo run --release --example make_levels`) holds 21 start states, 17 train and 4 held out (ids 4, 9, 14, 19). Levels are reached by teleporting the overworld map cursor (`$1F17`/`$1F19`) and pressing A; writing `$13BF`/level numbers does nothing. `make_levels` scans the map, drops duplicate placeholder levels (first-frame hash), vertical levels and levels a random policy cannot progress 400 px in. Known odd levels: tl 0x3b (lava cave, near-impossible), tl 0x3f/0x45 (door-ended castle rooms, the door leads to another area), tl 0x54 underwater and tl 0x39 dark (random policy stalls). Original level 1 (tl 0x2a) is id 7.
- **Training features (all behind flags, see `src/bin/train.rs` header):** PLR level sampler (`src/plr.rs`, score = mean |GAE| per level), IMPALA-CNN trunk (`TRUNK`, `WIDTH`), entity vector from RAM fused with the CNN (`ENTITIES`, `src/entities.rs`), DrAC random-shift augmentation (`AUG_PAD`, `AUG_COEF`), speed-scaled clear bonus (`SPEED_K`, `src/reward.rs`; episode cap derived from the in-game timer, 1 timer tick = 41 frames), held-out evaluation every `EVAL_EVERY` iterations. Checkpoints save `net.cfg` so `eval` rebuilds the right architecture; `eval` takes `LEVEL=<id>`. Old single-level checkpoints are incompatible.
- **Core/state compatibility:** `levels/*.state` only load on the snes9x build that made them. The VM's apt core (1.53, 2016) rejects states from the 2026-09 pacman core (`retro_unserialize failed`); on the VM use the libretro nightly core (`CORE=/root/cores/snes9x_libretro.so`) or regenerate `levels/` there.
- **IMPALA cost on a T4:** the trunk is 3 stages of conv3x3+ReLU+maxpool (no residual blocks; the 15-conv version cost 2.4 ms/sample fwd+bwd vs 0.2 for Nature, now 1.08). WIDTH=2 is ~2.9 ms/sample.
- **Clear detection:** goal timer `$1493 != 0` only. Doors and pipes leave level mode but are not wins.
- **Gotcha:** a process can hold one live emulator per core copy; dropping two `VecEnv`s corrupts the heap at exit, so `train` ends with `process::exit(0)` after saving.
- **Difficulty:** a biased-random policy (mostly run+jump right) clears the level about 1 episode in 4, in about 650 agent steps. Expect fast learning; if training stalls, suspect a bug before the algorithm.
- **Buttons:** B jumps, Y runs (SNES layout). A is spin jump and is not in the action set.
- **Frame skip** 4 and **frame stack** 4, grayscale, downscaled (about 84x84).
- **Discrete action set** of a few button combos (right, right+B run, right+A jump, right+B+A, none). Keep it small.
- **Reward** from RAM, not pixels: progress in X position per step, a bonus on level clear, a penalty on death, a small time penalty. Reward shaping drives sample efficiency.
- RAM addresses to confirm against a SMW RAM map before use: X position (`$7E0094`, 16 bit), player animation state (`$7E0071`, death state), end-level timer (`$7E1493`), lives (`$7E0DBE`).
- Algorithm: **PPO** with parallel environments. DQN is the alternative if PPO stalls.

## Infinite lives

Probably unnecessary: reset from a save state handles death. If lives still matter, the easy route is **writing the lives RAM byte every frame** through the emulator, with no ROM edit. Otherwise, a Game Genie or Pro Action Replay code for infinite lives works in most SNES emulators, and SMW ROM hacks are catalogued at smwcentral.net. Do not touch the ROM until RAM writes prove insufficient.

## Seeing the result

- During training: log episode return and max X reached; save a checkpoint of the best policy.
- Final video: a script that loads the best checkpoint, runs one deterministic episode, writes frames to MP4 with ffmpeg. Runs headless on Colab, so the video can be downloaded.
- Live viewing is optional and secondary; the recorded MP4 is the deliverable.

## Working in this repo

- Rust edition 2024. `cargo check` and `cargo test` run locally before anything goes to Colab.
- Test the `Env` against a fake environment so the training loop is verifiable without a ROM.
- Commit small steps; the repo is new and has no commits yet.

## Results so far (2026-10-05)

- **Baseline** (Nature CNN, 1M steps, 4 epochs, 16 envs): clears 3 of 17 training levels (L7 = original level 1 at 23%, L17 20%, L12 13%) plus the trivial L15; 0/30 on every held-out level. Level ids are in `levels/manifest.txt`.
- **Full model** (IMPALA + entities + PLR + augmentation) was aborted at ~450k steps: ~140 steps/s on the T4 (GPU saturated), and PLR put ~80% of the weight on one level at a time. Never compared. Revisit with a softer `PLR_TEMP`/`PLR_RHO` before trusting PLR.
- **Long run** (Nature, warm start from the baseline, `ENVS=64 EPOCHS=2 PLR=0 EXCLUDE=8`, ~805 steps/s), stopped at 5.73M of a planned 30M steps: 20-episode local eval gives L7 60%, L12 70%, L17 100%, L15 100%; every other training level 0%. **No transfer:** held-out L9 and L14 clear 0/20 (the in-training held-out clears of ~5-15% were noise from 8-16 episodes per eval; use 30+ episodes per level, e.g. the `eval` binary per level).
- The 0% levels fail by dying at the same x every time at a pit/gap (L1, L3) or a grinder section (L13): a jump-skill problem, not a missing mechanic. Needs more training.
- Checkpoints, log, and clips: `runs/long_5M/` (checkpoint, `SUMMARY.md`), `runs/videos/` (long-run clips). Not committed.
- Speed notes: IMPALA is ~5x Nature per update on the T4 even after dropping the residual blocks; with 64 envs the loop is update-bound (rollout 4.6 s, update 5.5 s per 8192 steps). Overlapping rollout and update would be the next throughput win. `EXCLUDE=<ids>` drops levels from the training pool.
- Resume: `INIT=runs/long_5M/latest TRUNK=nature ENTITIES=0 AUG_PAD=0 PLR=0 EXCLUDE=8 ENVS=64 EPOCHS=2`, on a core matching `levels/`.
