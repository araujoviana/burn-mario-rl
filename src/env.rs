//! Super Mario World level-1 environment: boot script, observations, actions, reward.

use crate::emulator::{Emulator, button};
use std::path::Path;

pub const OBS_W: usize = 84;
pub const OBS_H: usize = 84;
pub const STACK: usize = 4;
pub const FRAME_SKIP: u32 = 4;
pub const OBS_LEN: usize = STACK * OBS_W * OBS_H;

/// Joypad masks the agent chooses from. B jumps, Y runs.
pub const ACTIONS: [u16; 7] = [
    0,
    button::RIGHT,
    button::RIGHT | button::B,
    button::RIGHT | button::Y,
    button::RIGHT | button::Y | button::B,
    button::B,
    button::LEFT,
];

// Work RAM offsets ($7E0000 + offset).
const RAM_GAME_MODE: usize = 0x100;
const RAM_PLAYER_STATE: usize = 0x71;
const RAM_PLAYER_X: usize = 0x94;
const RAM_END_LEVEL_TIMER: usize = 0x1493;

const MODE_OVERWORLD: u8 = 0x0E;
const MODE_IN_LEVEL: u8 = 0x14;
const PLAYER_DYING: u8 = 9;

const REWARD_PER_TILE: f32 = 1.0 / 16.0;
const REWARD_CLEAR: f32 = 100.0;
const PENALTY_DEATH: f32 = 20.0;
const PENALTY_STEP: f32 = 0.02;
const MAX_STEPS: u32 = 1500;
/// Steps allowed without a new furthest X before the episode is cut.
const MAX_STALL: u32 = 200;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    Running,
    Died,
    Cleared,
    /// Out of steps, or no progress for too long.
    Timeout,
}

pub struct StepResult {
    pub reward: f32,
    pub outcome: Outcome,
}

fn x_pos(emu: &Emulator) -> u16 {
    let r = emu.ram();
    u16::from_le_bytes([r[RAM_PLAYER_X], r[RAM_PLAYER_X + 1]])
}

fn game_mode(emu: &Emulator) -> u8 {
    emu.ram()[RAM_GAME_MODE]
}

fn hold(emu: &mut Emulator, mask: u16, frames: u32) {
    emu.set_buttons(mask);
    for _ in 0..frames {
        emu.run_frame();
    }
}

/// Hold `mask` (or tap it when `tap`) until `done` is true, failing after `max` frames.
fn drive(emu: &mut Emulator, mask: u16, tap: bool, max: u32, done: impl Fn(&Emulator) -> bool) -> Result<(), String> {
    for f in 0..max {
        if done(emu) {
            emu.set_buttons(0);
            return Ok(());
        }
        emu.set_buttons(if !tap || f % 60 < 5 { mask } else { 0 });
        emu.run_frame();
    }
    Err(format!("boot script timed out (mode {:#04x})", game_mode(emu)))
}

/// Play the menus from power-on to the first frame of the first level and return a save state there.
pub fn level_start_state(emu: &mut Emulator) -> Result<Vec<u8>, String> {
    hold(emu, 0, 600); // ROM intro is black for ~5 s
    drive(emu, button::START, true, 3000, |e| game_mode(e) == MODE_OVERWORLD)?; // title -> intro -> map
    hold(emu, 0, 60);
    drive(emu, button::A, true, 600, |e| game_mode(e) == MODE_IN_LEVEL)?; // enter Yoshi's House
    hold(emu, 0, 30);
    drive(emu, button::RIGHT, false, 1200, |e| game_mode(e) == MODE_OVERWORLD)?; // walk out of the house
    hold(emu, 0, 300); // path to the next node opens
    hold(emu, button::RIGHT, 40); // walk to the first level
    hold(emu, 0, 200);
    drive(emu, button::A, true, 600, |e| game_mode(e) == MODE_IN_LEVEL)?;
    hold(emu, 0, 60); // let the level fade in and Mario land
    emu.save_state()
}

fn grayscale_downscale(emu: &Emulator, out: &mut [u8]) {
    emu.with_frame(|rgb, w, h| {
        let (w, h) = (w as usize, h as usize);
        for oy in 0..OBS_H {
            let (y0, y1) = (oy * h / OBS_H, ((oy + 1) * h).div_ceil(OBS_H));
            for ox in 0..OBS_W {
                let (x0, x1) = (ox * w / OBS_W, ((ox + 1) * w).div_ceil(OBS_W));
                let mut sum = 0u32;
                for y in y0..y1 {
                    for x in x0..x1 {
                        let p = &rgb[(y * w + x) * 3..][..3];
                        sum += (77 * p[0] as u32 + 150 * p[1] as u32 + 29 * p[2] as u32) >> 8;
                    }
                }
                out[oy * OBS_W + ox] = (sum / ((y1 - y0) * (x1 - x0)) as u32) as u8;
            }
        }
    });
}

pub struct MarioEnv {
    emu: Emulator,
    start_state: Vec<u8>,
    /// `STACK` grayscale planes, oldest first.
    obs: Vec<u8>,
    prev_x: u16,
    max_x: u16,
    steps: u32,
    stall: u32,
}

impl MarioEnv {
    pub fn new(core: &Path, rom: &Path, start_state: Vec<u8>) -> Result<Self, String> {
        let mut env = Self { emu: Emulator::load(core, rom)?, start_state, obs: vec![0; OBS_LEN], prev_x: 0, max_x: 0, steps: 0, stall: 0 };
        env.reset()?;
        Ok(env)
    }

    pub fn emulator(&self) -> &Emulator {
        &self.emu
    }

    pub fn reset(&mut self) -> Result<&[u8], String> {
        self.emu.load_state(&self.start_state)?;
        self.emu.set_buttons(0);
        self.emu.run_frame(); // refresh the framebuffer after the state load
        self.prev_x = x_pos(&self.emu);
        self.max_x = self.prev_x;
        self.steps = 0;
        self.stall = 0;
        let plane = OBS_W * OBS_H;
        grayscale_downscale(&self.emu, &mut self.obs[..plane]);
        for i in 1..STACK {
            self.obs.copy_within(..plane, i * plane);
        }
        Ok(&self.obs)
    }

    pub fn observation(&self) -> &[u8] {
        &self.obs
    }

    pub fn step(&mut self, action: usize) -> StepResult {
        self.emu.set_buttons(ACTIONS[action]);
        for _ in 0..FRAME_SKIP {
            self.emu.run_frame();
        }
        let plane = OBS_W * OBS_H;
        self.obs.copy_within(plane.., 0);
        grayscale_downscale(&self.emu, &mut self.obs[(STACK - 1) * plane..]);

        let ram = self.emu.ram();
        let dying = ram[RAM_PLAYER_STATE] == PLAYER_DYING;
        let cleared = ram[RAM_END_LEVEL_TIMER] != 0 || (game_mode(&self.emu) != MODE_IN_LEVEL && !dying);
        let x = x_pos(&self.emu);

        self.steps += 1;
        let dx = (x as i32 - self.prev_x as i32).clamp(-30, 30) as f32;
        self.prev_x = x;
        if x > self.max_x {
            self.max_x = x;
            self.stall = 0;
        } else {
            self.stall += 1;
        }

        let mut reward = dx * REWARD_PER_TILE - PENALTY_STEP;
        let outcome = if cleared {
            reward += REWARD_CLEAR;
            Outcome::Cleared
        } else if dying {
            reward -= PENALTY_DEATH;
            Outcome::Died
        } else if self.steps >= MAX_STEPS || self.stall >= MAX_STALL {
            Outcome::Timeout
        } else {
            Outcome::Running
        };
        StepResult { reward, outcome }
    }
}
