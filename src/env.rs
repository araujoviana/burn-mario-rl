//! Super Mario World environment: boot script, level selection, observations, actions, reward.

use crate::emulator::{Emulator, button};
use crate::entities::{ENT_LEN, entity_vector};
use crate::levels::{Level, LevelSet};
use std::path::Path;
use std::sync::Arc;

pub const OBS_W: usize = 84;
pub const OBS_H: usize = 84;
pub const STACK: usize = 4;
/// Default emulator frames per agent step. Override with the `FRAME_SKIP` env var (read once; see `frame_skip`).
pub const FRAME_SKIP: u32 = 4;

pub fn frame_skip() -> u32 {
    static N: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *N.get_or_init(|| std::env::var("FRAME_SKIP").ok().and_then(|v| v.parse().ok()).filter(|&n| n >= 1).unwrap_or(FRAME_SKIP))
}
pub const OBS_LEN: usize = STACK * OBS_W * OBS_H;

/// Joypad masks the agent chooses from. B jumps, Y runs, A is a spin jump.
#[cfg(not(feature = "extended_actions"))]
pub const ACTIONS: [u16; 7] = [
    0,
    button::RIGHT,
    button::RIGHT | button::B,
    button::RIGHT | button::Y,
    button::RIGHT | button::Y | button::B,
    button::B,
    button::LEFT,
];

/// The seven basic actions plus down (enter pipes, duck), up (climb vines, enter doors) and spin jumps.
#[cfg(feature = "extended_actions")]
pub const ACTIONS: [u16; 11] = [
    0,
    button::RIGHT,
    button::RIGHT | button::B,
    button::RIGHT | button::Y,
    button::RIGHT | button::Y | button::B,
    button::B,
    button::LEFT,
    button::DOWN,
    button::UP,
    button::RIGHT | button::A,
    button::RIGHT | button::Y | button::A,
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
const PENALTY_DEATH: f32 = 20.0;

/// Penalty when an episode ends by stalling or running out of time (env `TIMEOUT_PENALTY`, default 15).
/// Without it, standing still at a pit edge beats risking the jump (death costs 20).
fn timeout_penalty() -> f32 {
    static P: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *P.get_or_init(|| std::env::var("TIMEOUT_PENALTY").ok().and_then(|v| v.parse().ok()).unwrap_or(15.0))
}
const PENALTY_STEP: f32 = 0.02;
/// A drop in X bigger than this in one step means a door or pipe took Mario to another area.
const AREA_CHANGE_DX: i32 = 200;
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

/// What killed Mario, judged from RAM on the first frame of the death.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum DeathCause {
    #[default]
    None,
    /// Fell below the screen.
    Pit,
    /// A hostile sprite was touching or next to Mario.
    Enemy,
    /// Anything else (spikes, lava, crushed, timer).
    Other,
}

impl DeathCause {
    pub fn name(self) -> &'static str {
        match self {
            DeathCause::None => "-",
            DeathCause::Pit => "pit",
            DeathCause::Enemy => "enemy",
            DeathCause::Other => "other",
        }
    }
}

/// Per-episode diagnostics: reward split by source and where and why the episode ended.
#[derive(Clone, Copy, Debug, Default)]
pub struct EpisodeTelemetry {
    pub r_progress: f32,
    pub r_coin: f32,
    pub r_time: f32,
    pub r_death: f32,
    pub r_timeout: f32,
    pub r_clear: f32,
    /// Bonus for discovering a new archive cell (added by the worker).
    pub r_explore: f32,
    pub end_x: u16,
    pub end_y: u16,
    pub cause: DeathCause,
}

fn classify_death(ram: &[u8]) -> DeathCause {
    let (mx, my) = (crate::ram::u16_at(ram, crate::ram::PLAYER_X) as i32, crate::ram::u16_at(ram, crate::ram::PLAYER_Y) as i32);
    for i in 0..crate::ram::SPRITE_SLOTS {
        if ram[crate::ram::SPRITE_STATUS + i] < 8 {
            continue;
        }
        let sx = ram[crate::ram::SPRITE_X_LO + i] as i32 | (ram[crate::ram::SPRITE_X_HI + i] as i32) << 8;
        let sy = ram[crate::ram::SPRITE_Y_LO + i] as i32 | (ram[crate::ram::SPRITE_Y_HI + i] as i32) << 8;
        if (sx - mx).abs() <= 24 && (sy - my).abs() <= 32 && !crate::grid::is_friendly(ram[crate::ram::SPRITE_TYPE + i]) {
            return DeathCause::Enemy;
        }
    }
    let camera_y = crate::ram::u16_at(ram, 0x1C) as i32;
    if my - camera_y >= 192 { DeathCause::Pit } else { DeathCause::Other }
}

pub struct StepResult {
    pub reward: f32,
    pub outcome: Outcome,
}

fn x_pos(emu: &Emulator) -> u16 {
    let r = emu.ram();
    u16::from_le_bytes([r[RAM_PLAYER_X], r[RAM_PLAYER_X + 1]])
}

pub(crate) fn game_mode(emu: &Emulator) -> u8 {
    emu.ram()[RAM_GAME_MODE]
}

pub(crate) fn hold(emu: &mut Emulator, mask: u16, frames: u32) {
    emu.set_buttons(mask);
    for _ in 0..frames {
        emu.run_frame();
    }
}

/// Hold `mask` (or tap it when `tap`) until `done` is true, failing after `max` frames.
pub(crate) fn drive(emu: &mut Emulator, mask: u16, tap: bool, max: u32, done: impl Fn(&Emulator) -> bool) -> Result<(), String> {
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

/// Press A until the game enters a level.
pub(crate) fn drive_to_level(emu: &mut Emulator, max: u32) -> Result<(), String> {
    drive(emu, button::A, true, max, |e| game_mode(e) == MODE_IN_LEVEL)
}

/// Play the menus from power-on to the first frame of the first level and return a save state there.
pub fn level_start_state(emu: &mut Emulator) -> Result<Vec<u8>, String> {
    boot_to_overworld(emu)?;
    drive_to_level(emu, 600)?;
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
    levels: Arc<LevelSet>,
    /// Index into `levels.levels` of the current episode.
    level: usize,
    /// Weight of the speed term in the clear bonus (see `reward::clear_bonus`).
    k_speed: f32,
    /// Reward per coin collected (env `COIN_K`); small next to progress and the clear bonus.
    coin_k: f32,
    prev_coins: u8,
    tele: EpisodeTelemetry,
    timer_start: u32,
    step_cap: u32,
    /// `STACK` grayscale planes, oldest first.
    obs: Vec<u8>,
    /// Entity vector of the latest frame (see `entities`).
    ent: Vec<f32>,
    prev_x: u16,
    max_x: u16,
    steps: u32,
    stall: u32,
}

impl MarioEnv {
    /// Single-level env from one start state.
    pub fn new(core: &Path, rom: &Path, start_state: Vec<u8>) -> Result<Self, String> {
        let set = LevelSet { levels: vec![Level { id: 0, translevel: 0, map_x: 0, map_y: 0, held_out: false, state: start_state }] };
        Self::with_levels(core, rom, Arc::new(set), 1.0)
    }

    /// Multi-level env; `reset_to` picks the level of each episode.
    pub fn with_levels(core: &Path, rom: &Path, levels: Arc<LevelSet>, k_speed: f32) -> Result<Self, String> {
        let mut env = Self {
            emu: Emulator::load(core, rom)?,
            levels,
            level: 0,
            k_speed,
            coin_k: std::env::var("COIN_K").ok().and_then(|v| v.parse().ok()).unwrap_or(0.3),
            prev_coins: 0,
            tele: EpisodeTelemetry::default(),
            timer_start: 0,
            step_cap: 0,
            obs: vec![0; crate::obs::len()],
            ent: vec![0.0; ENT_LEN],
            prev_x: 0,
            max_x: 0,
            steps: 0,
            stall: 0,
        };
        env.reset_to(0)?;
        Ok(env)
    }

    /// Index of the level the current episode runs on.
    pub fn level(&self) -> usize {
        self.level
    }

    /// Mario's Y position after the latest step.
    pub fn y(&self) -> u16 {
        crate::ram::u16_at(self.emu.ram(), crate::ram::PLAYER_Y)
    }

    /// Records a bonus reward the caller adds to the step (shows up in the telemetry split).
    pub fn credit_exploration(&mut self, bonus: f32) {
        self.tele.r_explore += bonus;
    }

    /// Mario's X position after the latest step.
    pub fn x(&self) -> u16 {
        self.prev_x
    }

    pub fn emulator(&self) -> &Emulator {
        &self.emu
    }

    /// Restart the current level.
    pub fn reset(&mut self) -> Result<&[u8], String> {
        self.reset_to(self.level)
    }

    /// Start a new episode on `level` (an index into the level set).
    pub fn reset_to(&mut self, level: usize) -> Result<&[u8], String> {
        self.reset_from(level, None)
    }

    /// Like `reset_to`, but starts from `state` (a snapshot taken on `level`) when given.
    pub fn reset_from(&mut self, level: usize, state: Option<&[u8]>) -> Result<&[u8], String> {
        self.level = level;
        self.emu.load_state(state.unwrap_or(&self.levels.levels[level].state))?;
        self.emu.set_buttons(0);
        self.emu.run_frame(); // refresh the framebuffer after the state load
        self.prev_x = x_pos(&self.emu);
        self.prev_coins = self.emu.ram()[crate::ram::COINS];
        self.tele = EpisodeTelemetry::default();
        self.max_x = self.prev_x;
        self.steps = 0;
        self.stall = 0;
        self.timer_start = crate::ram::timer(self.emu.ram());
        self.step_cap = crate::reward::step_cap(self.timer_start);
        if crate::obs::kind() == crate::obs::ObsKind::Grid {
            self.obs.fill(0);
            crate::grid::build(self.emu.ram(), &mut self.obs);
            crate::grid::build(self.emu.ram(), &mut self.obs); // second call copies the sprite planes into the history planes
        } else {
            let plane = OBS_W * OBS_H;
            grayscale_downscale(&self.emu, &mut self.obs[..plane]);
            for i in 1..STACK {
                self.obs.copy_within(..plane, i * plane);
            }
        }
        entity_vector(self.emu.ram(), &mut self.ent);
        Ok(&self.obs)
    }

    /// Snapshot of the emulator, for the frontier archive.
    pub fn snapshot(&self) -> Result<Vec<u8>, String> {
        self.emu.save_state()
    }

    /// True when Mario stands on something and is not dying.
    pub fn grounded(&self) -> bool {
        let ram = self.emu.ram();
        ram[crate::ram::PLAYER_BLOCKED] & 4 != 0 && ram[RAM_PLAYER_STATE] == 0
    }

    /// Diagnostics of the episode in progress (final once it has ended).
    pub fn telemetry(&self) -> &EpisodeTelemetry {
        &self.tele
    }

    pub fn max_x(&self) -> u16 {
        self.max_x
    }

    pub fn observation(&self) -> &[u8] {
        &self.obs
    }

    pub fn entities(&self) -> &[f32] {
        &self.ent
    }

    pub fn step(&mut self, action: usize) -> StepResult {
        self.step_with(action, |_| {})
    }

    /// Like `step`, but calls `on_frame` after every emulator frame (for recording video).
    pub fn step_with(&mut self, action: usize, mut on_frame: impl FnMut(&Emulator)) -> StepResult {
        self.emu.set_buttons(ACTIONS[action]);
        for _ in 0..frame_skip() {
            self.emu.run_frame();
            on_frame(&self.emu);
        }
        if crate::obs::kind() == crate::obs::ObsKind::Grid {
            crate::grid::build(self.emu.ram(), &mut self.obs);
        } else {
            let plane = OBS_W * OBS_H;
            self.obs.copy_within(plane.., 0);
            grayscale_downscale(&self.emu, &mut self.obs[(STACK - 1) * plane..]);
        }
        entity_vector(self.emu.ram(), &mut self.ent);

        let ram = self.emu.ram();
        let dying = ram[RAM_PLAYER_STATE] == PLAYER_DYING;
        // Goal timer only: doors and pipes also leave level mode but are not wins.
        let cleared = ram[RAM_END_LEVEL_TIMER] != 0;
        let x = x_pos(&self.emu);

        self.steps += 1;
        let raw_dx = x as i32 - self.prev_x as i32;
        let dx = raw_dx.clamp(-30, 30) as f32;
        self.prev_x = x;
        if raw_dx < -AREA_CHANGE_DX {
            self.max_x = x; // new area: progress is measured from here
            self.stall = 0;
        } else if x > self.max_x {
            self.max_x = x;
            self.stall = 0;
        } else {
            self.stall += 1;
        }

        let mut reward = dx * REWARD_PER_TILE - PENALTY_STEP;
        self.tele.r_progress += dx * REWARD_PER_TILE;
        self.tele.r_time -= PENALTY_STEP;
        let coins = self.emu.ram()[crate::ram::COINS];
        // the counter wraps 99 -> 0 (and gives a 1UP), so take the difference modulo 100
        let coin_reward = self.coin_k * ((coins as i32 - self.prev_coins as i32).rem_euclid(100)) as f32;
        reward += coin_reward;
        self.tele.r_coin += coin_reward;
        self.prev_coins = coins;
        let outcome = if cleared {
            let bonus = crate::reward::clear_bonus(crate::ram::timer(self.emu.ram()), self.timer_start, self.k_speed);
            reward += bonus;
            self.tele.r_clear = bonus;
            Outcome::Cleared
        } else if dying {
            reward -= PENALTY_DEATH;
            self.tele.r_death = -PENALTY_DEATH;
            self.tele.cause = classify_death(self.emu.ram());
            Outcome::Died
        } else if self.steps >= self.step_cap || self.stall * frame_skip() >= MAX_STALL * FRAME_SKIP {
            reward -= timeout_penalty();
            self.tele.r_timeout = -timeout_penalty();
            Outcome::Timeout
        } else {
            Outcome::Running
        };
        self.tele.end_x = x;
        self.tele.end_y = crate::ram::u16_at(self.emu.ram(), crate::ram::PLAYER_Y);
        StepResult { reward, outcome }
    }
}
