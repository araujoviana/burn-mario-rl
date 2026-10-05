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

## Update 2026-10-05 (later): grid observation, frontier archive, telemetry

- **Pool correction:** L12 and L17 are switch palaces (the "clear" is touching a `!` switch) and L15 is a stub level (goal gate next to the start). `train` now drops `8,12,15,17` by default (`EXCLUDE`). Before this, only L7 (original level 1) was a real clear.
- **Tile-grid observation** (`TRUNK=gridmlp`, `src/grid.rs`, `src/obs.rs`): a 7x14x20 grid read from RAM (Map16 tiles at `$7EC800`/`$7FC800`: `0x25` empty, ids `>= 0x100` solid, `0x2B` coin; sprites split hostile/friendly, plus last frame's sprite planes) fused with the entity vector. About 30x cheaper to update than the Nature CNN on a CPU, and it learns much faster (L7 about 80% in under 1M steps; the pixel model needed 5.7M). Vertical levels give an empty grid. `TRUNK=grid` is a conv version, only 3x cheaper on the CPU backend. Pixel and grid checkpoints are not interchangeable (`net.cfg` records the trunk).
- **Frontier archive** (`src/archive.rs`, `FRONTIER=<share>`, `EXPLORE_BONUS`): save states stored the first time Mario stands on ground in each (96 px x 64 px) cell of a level; a share of new episodes starts from one (rarely tried cells and the two furthest-right cells favoured), and discovering a cell pays a bonus. Only from-start episodes count toward the logged clear rates; the log also prints `frontier-start clear%`.
- **Win paths** (`PATH_SHARE`, default 0.25): every from-start win is replayed from the level start to check determinism (all replays so far clear again); the shortest per level is kept as a state every 8 steps and its action list is written to `<run>/wins/level<i>_<n>steps.actions`. A share of episodes starts on that path, nearest the finish first, and the start moves back when the agent clears from there more than 70% of the time. Starts are split path 25% / frontier cells 50% / level start 25%. The log prints `win paths: L7:210steps/back100% ...`.
- **Stalling equilibrium:** with a 20-point death penalty and no timeout penalty, the agent learns to stand still at a pit edge (L1: 93-98% timeouts). `TIMEOUT_PENALTY` (default 15) charges stalls; 25 is being tested.
- **Telemetry:** every run writes `episodes.jsonl` (per episode: level, from-start flag, outcome, end x/y, death cause pit/enemy/other, reward split, action counts) and `telemetry.jsonl` (per iteration: losses, KL, clip fraction, explained variance, advantage stats, action mix). Summarise with `python3 scripts/analyze.py runs/<dir>`. `cargo run --release --example trace` (env CKPT, LEVEL) dumps every step's action probabilities, value and position and prints the tile grid the network saw. `examples/action_stats.rs` counts the action mix.
- **Findings:** pits kill on L1, L3, L13, L18 (same x every time); enemies on L0, L5 (moles), L6, L11; L20 needs a jump over an 8-tile pipe wall. The stuck levels need timing, routes that go up or left, waiting for platforms, and a switch-then-run. See `runs/analysis/stuck_levels.png`.
- **Action set:** `--features extended_actions` adds down, up and spin jump (11 actions). Tested 2026-10-05: slower to learn from the start than 7 actions and no clears on the pipe levels yet. Default is 7.
- **Other flags:** `FRAME_SKIP` (default 4; 2 was slower, 8 was clearly better in the 2026-10-05 runs, so use `FRAME_SKIP=8 GAMMA=0.98` and pass `FRAME_SKIP=8` to `eval` too; 12 is being tested), `GAMMA`, `REWARD_SCALE`. Checkpoints are written atomically (`<name>_tmp.mpk` then renamed) after a battery death left 0-byte files.
- **Compute:** laptop (12 cores, no CUDA) does about 700-1000 steps/s with the grid model; the VM (T4, 36 vCPU) about 1700-2800 steps/s per run, two runs side by side. VM access: `ssh mario-vm` (key and password in `~/.config/mario-rl/`, untracked). Stop the VM when idle.
- **Parked idea:** human demonstrations of the stuck levels, see `docs/ideas/2026-10-05-human-demonstrations.md`.

### Results of the 2026-10-05 grid runs (VM, 7 actions, from scratch)

- Frame skip 8 + gamma 0.98 + path curriculum (`runs/vm_ext8`, 3.5M steps): from the level start L7 94%, L6 35%, L0 29%, L16 9%, all other levels 0%. Frame skip 4 (`vm_ext7`) at 4.8M: L7 95%, L6 31%, L0 0%. Win paths recorded for L0, L6, L7, L16.
- 11 actions (`vm_ext5`) learned slower than 7 (L7 71%, L6 4% at 4.3M). Timeout penalty 25 vs 15 made little difference. L1 and L20 still stall (86-95% timeouts); L3, L10, L13, L18 die at the same pits.
- Held-out (20 attempts each, `vm_ext8` at 3.5M): L14 2/20, L4 0/20 (furthest x 1238), L9 0/20 (962), L19 0/20 (1405). First sign of transfer, but only 2 clears.
- Checkpoints and telemetry of every arm are in `runs/vm_ext<n>/`, logs in `runs/vm_logs/` (untracked). Videos of the levels beaten: `runs/videos/grid/levels_beaten.mp4`.

### Stop point (2026-10-05, afternoon): where to resume

- **Best checkpoint:** `runs/vm_ext8/stop_point.mpk` (frame skip 8, about 10.7M steps, `net.cfg` = `gridmlp 1 1`). From the level start: L7 97%, L0 87%, L6 84%, L10 38%, L16 39%; everything else 0%. Second arm `runs/vm_ext9/` (frame skip 12, 5.3M steps) is close behind. Win action lists (verified replays) are in `runs/vm_ext8/wins/` (24 files, L0, L6, L7, L10, L16) and `runs/vm_ext9/wins/`.
- **Resume ext8** (7 actions build, `cargo build --release --features cuda` on the VM or `--features flex` locally; levels and the ROM must be present; on the VM `CORE=/root/cores/snes9x_libretro.so`):
  `INIT=runs/vm_ext8/stop_point TRUNK=gridmlp ENTITIES=1 PLR=0 FRONTIER=0.5 PATH_SHARE=0.25 EXPLORE_BONUS=1.0 TIMEOUT_PENALTY=15 FRAME_SKIP=8 GAMMA=0.98 ENVS=30 ROLLOUT=128 EPOCHS=4 WINS_DIR=runs/vm_ext8/wins CKPT_DIR=runs/<new dir> ./train`.
  `WINS_DIR` makes the first worker replay the saved wins at startup and rebuild the win paths (the archive itself is in memory only). The path start fraction restarts at 10% and moves back quickly. The frontier cells refill within minutes.
- **Evaluate** a checkpoint with the same frame skip: `FRAME_SKIP=8 CKPT=<path without .mpk> LEVEL=<id> EPISODES=40 OUT=clip.mp4 eval`.
- **VM:** `mario-vm` was stopped (not deleted) at this point; the disk keeps `/root/proj`, `/root/target_new` (7 actions, CUDA) and `/root/target_ext` (11 actions). Start it again from the Huawei console or the API (`batch_start_servers`, region `la-south-2`, server id in the memory notes); ssh works with the saved key once it is up.
- **Next steps:** (1) resume ext8 and see whether L3, L13, L18 (pit deaths at the same x) get a first win; (2) held-out progress metric in the log (clears alone hide movement); (3) human demonstrations of the stuck levels, `docs/ideas/2026-10-05-human-demonstrations.md`; (4) possibly a recurrent policy for waiting and platform timing.
