# Idea: human demonstrations for the stuck levels

Status: parked by the user (they will record the demos later). Not implemented.

## Why
Seven training levels (L1, L2, L3, L5, L11, L13, L20) need behaviour that reward-driven exploration does not find:
precise jump timing (head bumps), a switch followed by a run before platforms vanish, routes that go up or left,
Monty Mole sections, and waiting on a line-guided platform. These are long exact sequences with a reward only at the end.

## What to build
1. **Recorder**: run the libretro host with keyboard or gamepad input at 60 fps, from a level's start state.
   Save, per frame: joypad mask; every 30-60 frames: a save state (`Emulator::save_state`) and Mario's x, y.
   Store under `demos/<level id>/<take>/` (gitignored: save states are large and core-version specific).
2. **Use the save states as frontier states**: load them into `archive::Archive` so training episodes start past
   each obstacle (this is the exploration fix, and it needs no policy change).
3. **Behaviour cloning**: (observation, action) pairs from the recorded frames, grouped into agent steps (4 frames),
   as an auxiliary cross-entropy loss during PPO (weight decaying to 0), or a pre-training phase before PPO.
4. **Evaluate** on the levels the demos cover and on the held-out levels, to see whether demos transfer.

## Notes
- Demonstrate each stuck level 3-5 times, including mistakes and recoveries from different positions.
- The action set must be able to express the demo (use the `extended_actions` feature: down, up, spin jump).
- A related idea from the same discussion: a vision-language model queried with a screenshot, used offline to
  write per-level waypoints or hints. Too slow for the training loop; plan separately.
