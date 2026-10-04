# Generalization across levels: design

Date: 2026-10-04. Status: approved in chat, pending review of this file.

## Problem

The agent trained on one level (level 1, plus curriculum start states from that same level) and
overfits to its layout. Goal: an agent that clears many Super Mario World levels, including levels
it never trained on, and that clears them fast.

## Decisions (from brainstorming)

- Level pool: 15+ levels, 3-4 held out for a generalization score.
- Level start states: made by RAM warp from the overworld, no per-level menu scripting.
- Entity RAM enters the network as a vector branch fused with the CNN features.
- Build staged, every piece behind a config flag so each can be ablated.
- Speed is rewarded explicitly (see Reward).

## Build order

0. **Spike: RAM warp.** Riskiest step. From an overworld save state, write the translevel number
   and game mode, let the level load, save the state. Verify by reproducing the hand-booted level 1
   state and comparing RAM. Fallback if it fails: scripted overworld walks, fewer levels.
1. Multi-level env and reward.
2. PLR sampler.
3. IMPALA trunk.
4. Entity vector and augmentation.
5. Evaluation and logging.

## 1. Multi-level env (`levels.rs`, `env.rs`, `vec_env.rs`)

- `LevelSet` holds `{id, translevel, state bytes, split: train | held_out}`.
- Level choice is per episode, not per worker. The sampler sits behind `Arc<Mutex<..>>`; each worker
  asks it for a level on every `reset`.
- Each step reply carries `level_id` so the learner can attribute rollout data to levels.
- Pool is limited to horizontal levels with a goal gate. Vertical, water-scroll and boss levels are
  filtered out using the layout flag in RAM, because the reward uses X progress.
- Reset and clear detection keep the existing logic (`$1493 != 0`, player state `$71 == 9`).

### Reward

- Progress `dx / 16` per step and the death penalty stay as they are.
- Step penalty stays at `0.02`. Raising it teaches the agent to die early.
- Clear bonus becomes `100 * (1 + k * timer_left_fraction)`, from the in-game timer in RAM (address
  to be confirmed against an SMW RAM map before use). `k` is config, start at about 1.
- `MAX_STEPS` becomes a per-level budget derived from the in-game timer, replacing the fixed 1500.
- Why: total progress is the same whether Mario walks or runs, so speed is only weakly rewarded
  today. Scaling by the timer works for short and long levels alike.

## 2. PLR (`plr.rs`)

- Per-level score is the mean |GAE| over that level's episodes, updated after each rollout, with an
  EMA.
- Sampling probability is `(1 - rho) * rank-based score weight + rho * staleness`. Config:
  temperature, `rho`, EMA rate.
- A flag switches it off for uniform sampling, as the baseline.
- Tests need no ROM: fake scores, check the sampling distribution and staleness behavior.

## 3. IMPALA trunk (`ppo.rs`)

- Three stages of conv 3x3, max-pool 3/2, then two residual blocks. Channels 16/32/32, width
  multiplier in config. Output feeds a 256-unit FC.
- Trunk selectable (IMPALA or the current Nature-DQN) for comparison.
- Measure cost on the T4 with `bench_net` before committing to a width.

## 4. Observations

- Entity vector: 12 sprite slots (type id, x and y relative to Mario, status) plus Mario's speed,
  power-up and on-ground flag. An MLP encodes it, and the result is concatenated with the CNN
  features before the actor and critic heads. Read from RAM, so level-agnostic.
- Augmentation: random shift (pad 4, random crop), DrAC style. PPO loss stays on clean frames; a
  KL term and a value-consistency term run on the augmented frames. Costs a second forward pass.
  Behind a flag.
- Frame stack stays at 4.

## 5. Evaluation and logging

- Training log shows per-level clear rate and held-out clear rate every N updates.
- Clear time in steps is logged next to clear rate, to show speed improving.
- `eval` takes a level id.

## Consequences

- The new trunk and inputs make old checkpoints (`vmD_best`) unusable. Training restarts from
  scratch and needs more steps. Level 1 curriculum start states become unnecessary.
- Training on the stopped Huawei VM needs the user's OK to start it. Never delete it without asking.
- CLAUDE.md lists other levels and generalization as out of scope; update it when work starts.

## Risks

- RAM warp may not load levels cleanly (spike decides).
- The horizontal-goal filter may leave fewer than 15 usable levels.
- Many new things at once; flags and the staged order keep failures attributable.
