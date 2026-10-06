# burn-mario-rl

A PPO agent that plays Super Mario World, written in Rust with [Burn](https://burn.dev). It runs the game in an
emulator (a snes9x libretro core, loaded through a small hand-written host in `src/emulator.rs`) and chooses
buttons from the game's memory instead of the screen.

The trained model clears 6 of the 8 levels it was trained to finish, from the level start, in 85 to 100 percent of
attempts. It does not transfer to levels it has not seen.

## Results

Included checkpoint (`models/mario_grid.mpk`), 40 attempts per level from the level start, frame skip 8, sampled actions:

| Level id | Cleared |
|---|---|
| 7 | 40/40 |
| 16 | 39/40 |
| 0 | 38/40 |
| 6 | 38/40 |
| 10 | 37/40 |
| 20 | 34/40 |
| 2 | 0/40 |
| 3 | 0/40 |

Level ids are the indices in `levels/manifest.txt`, which `make_levels` writes from your own ROM. The pool has 21
levels. These 8 were the target set. The rest were dropped (switch palaces and a stub level count as a clear
without being one, and the others were not pursued).

Levels 2 and 3 stay at 0% from the start. Both have a human demonstration, and the agent clears them when started
partway through, but not from the beginning. Level 3 fails at the same pit in almost every attempt.

Held-out levels (4, 9, 14, 19, never trained on): 0 clears in 160 attempts with this checkpoint (an earlier,
shorter-trained checkpoint cleared level 14 in 2 of 20). The agent did not transfer to held-out
levels. The training levels are few and fixed, so memorizing where to jump is enough to clear them.

Training was about 50M agent steps in total, over several resumed runs, at roughly 1,400 steps/s per run on a 36-vCPU cloud VM
with one T4. Emulation on the CPU is the bottleneck.

## How it works

**Observation.** Each step the agent reads the Map16 tile map and the sprite table from work RAM and builds a
7 x 14 x 20 grid around Mario: solid tiles, coins, other tiles, hostile sprites, friendly sprites, and the sprite
positions from the previous step (so it can see motion). A 70-value vector of nearby entities and Mario's state is
added. A two-layer MLP reads both. This replaced a pixel CNN and made updates about 30 times cheaper on a CPU. Level 7 reached about 80% in under 1M
steps with the grid. The earlier pixel CNN needed 5.7M steps for the same level.

Known flaw: tile ids below `0x100` that are not coins are all labelled "other", which includes the dirt under the
ground. A thick wall looks the same as a flat floor to the network.

**Actions.** 7 button combinations (none, right, right+B, right+Y, right+Y+B, B, left). B jumps and Y runs. Frame
skip 8 with gamma 0.98 trained better than frame skip 4 or 2 in my runs (likely because a long jump takes fewer decisions).

**Reward.** Progress in x, a bonus for clearing, a penalty for dying, a penalty for timing out (without it the agent
learns to stand at the edge of a pit), a small coin reward and a small time penalty. All are in `src/env.rs`.

**Start states.** Training does not always start at the level start:
- A cell archive (`src/archive.rs`) saves an emulator state the first time Mario stands on the ground in each
  96 x 64 pixel cell of a level, and a share of episodes starts from those, with a bonus for discovering new cells
  (the idea from Go-Explore).
- Every win from the level start is replayed to check that the game is deterministic, and the shortest win per level
  is stored as a chain of saved states. Episodes start along that chain, closest to the finish first. When the agent
  clears from a start more than 70% of the time, the start moves back (the idea from OpenAI's "Learning Montezuma's
  Revenge from a Single Demonstration").
- A human run, recorded with `examples/play.rs`, is turned into the same kind of chain. That is what moved level 20
  from 0% to 85%.

**Telemetry.** Training writes `episodes.jsonl` (outcome, end position, death cause, reward split and action counts
per episode) and `telemetry.jsonl` (losses, KL, explained variance per iteration). `scripts/analyze.py` summarises a
run per level.

## Requirements

- Rust (edition 2024) and `ffmpeg` for videos.
- A Super Mario World (USA) ROM, placed in the repo root as `Super Mario World (USA).sfc`. It is not included.
- A snes9x libretro core. Set its path with `CORE` (default `/usr/lib/libretro/snes9x_libretro.so`). Save states only
  load on the core build that made them, so build `levels/` with the core you will train with.

## Usage

```sh
# Build the level set from your ROM (writes levels/*.state and levels/manifest.txt)
cargo run --release --example make_levels

# Play the trained model on one level and record the first clear
FRAME_SKIP=8 CKPT=models/mario_grid LEVEL=7 EPISODES=40 OUT=clip.mp4 cargo run --release --bin eval

# Train from scratch (CPU backend; use --features cuda for a GPU)
TRUNK=gridmlp ENTITIES=1 PLR=0 EXCLUDE=1,5,8,11,12,13,15,17,18 FRONTIER=0.5 PATH_SHARE=0.25 \
  FRAME_SKIP=8 GAMMA=0.98 ENVS=12 TOTAL_STEPS=20000000 CKPT_DIR=runs/mine \
  cargo run --release --features flex --bin train

# Record a demonstration (opens a window; arrow keys, X = jump, Z = run), then check that it replays
LEVEL=3 cargo run --release --features play --example play
cargo run --release --example demo_check
# Resume training with it: DEMOS_DIR=demos INIT=runs/mine/latest ... (see the header of src/bin/train.rs)
```

`eval` has to use the same `FRAME_SKIP` as the training run. All training options are environment variables, listed
at the top of `src/bin/train.rs`.

Other tools in `examples/`: `trace` (per-step action probabilities, value and the grid the network saw),
`ghosts` with `scripts/ghosts.py` (many attempts drawn on one level at once), `showcase` with `scripts/showcase.py`
(a clear with the tile map, action probabilities and value estimate beside the game; the on-screen text is in
Portuguese), `action_stats`, `death_frames`, `bench_net`, `env_check`, `vec_check`.

## Layout

- `src/emulator.rs`: libretro host (frames, input, RAM, save states).
- `src/env.rs`, `src/levels.rs`, `src/vec_env.rs`: environment, level set, parallel workers (one core copy per thread).
- `src/grid.rs`, `src/entities.rs`, `src/obs.rs`: observation.
- `src/ppo.rs`: networks and the PPO update. `src/archive.rs`: cell archive and winning paths.
- `src/bin/train.rs`, `src/bin/eval.rs`: training and evaluation.

## Credits

Burn, snes9x and libretro. The start-state ideas come from Go-Explore (Ecoffet et al.) and OpenAI's single
demonstration work on Montezuma's Revenge (Salimans and Chen).

MIT licence for the code. The ROM and the snes9x core are not part of this repository and have their own terms.
