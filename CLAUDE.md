# burn-mario-rl

Reinforcement learning agent that beats the first level of Super Mario World (SNES). Written in Rust with [Burn](https://burn.dev). The end product is a **video (or live run) of the agent clearing the level**, to post on LinkedIn.

## Goal and priorities

1. **Milestone 1:** the agent clears the first level (the one after Yoshi's House; the map labels it Yoshi's Island 1). Nothing else matters until this works.
2. **Deliverable:** a clean MP4 of a winning run, recorded at native resolution, upscaled for posting.
3. **Speed of learning:** the user wants wall-clock training to be short. Pick the fastest GPU Colab offers and favor sample-efficient choices over elegant ones.

Out of scope until Milestone 1 is done: other levels, generalization, multi-game support.

## Stack and fallback

- **Primary:** Rust + Burn.
- **Fallback:** Python (stable-retro + PyTorch / Stable-Baselines3). Switch only when the Rust path is blocked, most likely by emulator bindings. Record the blocker in this file when switching.
- **Training runs on a Colab VM with a GPU**, headless. Use the `colab-operator` skill for sessions, file sync, and shell on the VM. Local machine is for development and smoke tests.
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
