//! N environments, one worker thread each. Every worker loads its own copy of the core `.so`
//! because the core keeps global state, so a shared handle would be one emulator.
//! Workers pick the level of each new episode from a shared `LevelSampler`.

use crate::archive::Frontier;
use crate::entities::ENT_LEN;
use crate::env::{ACTIONS, EpisodeTelemetry, MarioEnv, Outcome, StepResult};
use crate::obs;
use crate::levels::LevelSet;
use crate::plr::SharedSampler;
use std::path::Path;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread::JoinHandle;

#[derive(Clone, Copy, Debug)]
pub struct EpisodeInfo {
    pub ret: f32,
    pub steps: u32,
    pub outcome: Outcome,
    pub max_x: u16,
    /// Index into the level set the episode ran on.
    pub level: usize,
    /// False when the episode began at an archived frontier state instead of the level start.
    pub from_start: bool,
    /// x Mario started the episode at (above the level start only for frontier episodes).
    pub start_x: u16,
    pub tele: EpisodeTelemetry,
    /// How often each action was chosen during the episode.
    pub actions: [u16; ACTIONS.len()],
}

struct Reply {
    id: usize,
    obs: Vec<u8>,
    ent: Vec<f32>,
    reward: f32,
    done: bool,
    info: Option<EpisodeInfo>,
    /// Level of the episode the env is in after this step (the new one when `done`).
    level: usize,
    /// Level the step itself was taken in.
    step_level: usize,
}

pub struct VecEnv {
    commands: Vec<Sender<usize>>,
    replies: Receiver<Reply>,
    workers: Vec<JoinHandle<()>>,
    /// Latest observation of every env, `n * obs::len()` bytes.
    pub obs: Vec<u8>,
    /// Latest entity vector of every env, `n * ENT_LEN` floats.
    pub ent: Vec<f32>,
    /// Level of the episode each env is currently in.
    pub levels: Vec<usize>,
    pub n: usize,
}

pub struct StepBatch {
    pub rewards: Vec<f32>,
    pub dones: Vec<bool>,
    /// Level each env was in when it took the step (differs from `VecEnv::levels` on episode ends).
    pub step_levels: Vec<usize>,
    pub finished: Vec<EpisodeInfo>,
}

/// Replays `trace` from the start of `level`; when it clears again, stores a state every `PATH_STRIDE` steps
/// as the level's winning path and writes the action list to `$CKPT_DIR/wins/`.
fn record_win(env: &mut MarioEnv, level: usize, trace: &[u8], fr: &Frontier) {
    if env.reset_from(level, None).is_err() {
        return;
    }
    let mut states = Vec::new();
    let mut outcome = Outcome::Running;
    for (i, &a) in trace.iter().enumerate() {
        if i % crate::archive::PATH_STRIDE == 0 {
            match env.snapshot() {
                Ok(s) => states.push(s),
                Err(_) => return,
            }
        }
        outcome = env.step(a as usize).outcome;
        if outcome != Outcome::Running {
            break;
        }
    }
    if outcome != Outcome::Cleared {
        eprintln!("win replay on level index {level} did not clear again ({} steps): not deterministic, path dropped", trace.len());
        return;
    }
    let dir = format!("{}/wins", std::env::var("CKPT_DIR").unwrap_or_else(|_| ".".into()));
    if std::fs::create_dir_all(&dir).is_ok() {
        let list: Vec<String> = trace.iter().map(|a| a.to_string()).collect();
        let _ = std::fs::write(format!("{dir}/level{level}_{}steps.actions", trace.len()), list.join(","));
    }
    eprintln!("win path recorded: level index {level}, {} steps, {} states", trace.len(), states.len());
    fr.archive.lock().expect("archive").offer_path(level, trace.len() as u32, states);
}

/// On resume: replays the shortest saved win of every level (`$WINS_DIR`, default `$CKPT_DIR/wins`) to rebuild
/// the win paths, which otherwise live only in memory. Files are named `level<index>_<n>steps.actions`.
fn load_saved_wins(env: &mut MarioEnv, fr: &Frontier) {
    let dir = std::env::var("WINS_DIR").unwrap_or_else(|_| format!("{}/wins", std::env::var("CKPT_DIR").unwrap_or_else(|_| ".".into())));
    let Ok(entries) = std::fs::read_dir(&dir) else { return };
    let mut best: std::collections::HashMap<usize, (usize, std::path::PathBuf)> = std::collections::HashMap::new();
    for e in entries.flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        let Some(rest) = name.strip_prefix("level").and_then(|r| r.strip_suffix("steps.actions")) else { continue };
        let Some((level, steps)) = rest.split_once('_') else { continue };
        if let (Ok(level), Ok(steps)) = (level.parse::<usize>(), steps.parse::<usize>()) {
            if best.get(&level).is_none_or(|(s, _)| steps < *s) {
                best.insert(level, (steps, e.path()));
            }
        }
    }
    for (level, (_, path)) in best {
        let Ok(text) = std::fs::read_to_string(&path) else { continue };
        let trace: Vec<u8> = text.trim().split(',').filter_map(|v| v.trim().parse().ok()).collect();
        if !trace.is_empty() && level < env.level_count() {
            record_win(env, level, &trace, fr);
        }
    }
}

/// Human demonstrations recorded with `examples/play.rs`: `$DEMOS_DIR` (default `demos`), files named
/// `level<id>_<frames>f.demo`. Each is replayed from the level start; a run that clears becomes the level's
/// winning path (unless a shorter win is already stored).
fn load_demos(env: &mut MarioEnv, fr: &Frontier) {
    let dir = std::env::var("DEMOS_DIR").unwrap_or_else(|_| "demos".into());
    let Ok(entries) = std::fs::read_dir(&dir) else { return };
    let stride = crate::archive::PATH_STRIDE * crate::env::frame_skip() as usize;
    for e in entries.flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        let Some(rest) = name.strip_prefix("level").and_then(|r| r.strip_suffix("f.demo")) else { continue };
        let Some((id, _frames)) = rest.split_once('_') else { continue };
        let Some(level) = id.parse::<usize>().ok().and_then(|id| env.level_index(id)) else { continue };
        let Ok(text) = std::fs::read_to_string(e.path()) else { continue };
        let masks: Vec<u16> = text.trim().split(',').filter_map(|v| v.trim().parse().ok()).collect();
        match env.replay_masks(level, &masks, stride) {
            Some(states) => {
                let steps = (masks.len() / crate::env::frame_skip() as usize) as u32;
                eprintln!("demo path loaded: {name}, {} frames, {} states", masks.len(), states.len());
                fr.archive.lock().expect("archive").offer_path(level, steps, states);
            }
            None => eprintln!("demo {name} did not clear when replayed (different core build?), skipped"),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn worker(id: usize, core: std::path::PathBuf, rom: std::path::PathBuf, levels: Arc<LevelSet>, sampler: SharedSampler, frontier: Option<Frontier>, k_speed: f32, seed: u64, commands: Receiver<usize>, replies: Sender<Reply>) {
    let mut rng = seed.wrapping_mul(0x9E3779B97F4A7C15) | 1;
    let mut env = match MarioEnv::with_levels(&core, &rom, levels, k_speed) {
        Ok(env) => env,
        Err(e) => {
            eprintln!("worker {id}: {e}");
            return;
        }
    };
    let _ = std::fs::remove_file(&core); // the library stays mapped after unlinking
    if id == 0 {
        if let Some(fr) = &frontier {
            load_saved_wins(&mut env, fr);
            load_demos(&mut env, fr);
        }
    }
    let first = sampler.lock().expect("sampler").sample(&mut rng);
    let _ = env.reset_to(first);
    let _ = replies.send(Reply { id, obs: env.observation().to_vec(), ent: env.entities().to_vec(), reward: 0.0, done: false, info: None, level: env.level(), step_level: env.level() });
    let (mut ret, mut steps) = (0.0f32, 0u32);
    let mut from_start = true;
    let mut start_x = env.x();
    let mut acts = [0u16; ACTIONS.len()];
    let mut trace: Vec<u8> = Vec::new(); // actions of the current episode when it began at the level start
    let mut from_path = false;
    while let Ok(action) = commands.recv() {
        let step_level = env.level();
        acts[action] = acts[action].saturating_add(1);
        if from_start {
            trace.push(action as u8);
        }
        let StepResult { mut reward, outcome } = env.step(action);
        let done = outcome != Outcome::Running;
        if let Some(fr) = &frontier {
            // Save a state the first time anyone stands on solid ground in a new cell, and pay a bonus for it.
            if !done && env.grounded() && fr.archive.lock().expect("archive").wants(step_level, env.x(), env.y()) {
                if let Ok(state) = env.snapshot() {
                    if fr.archive.lock().expect("archive").offer(step_level, env.x(), env.y(), state) {
                        reward += fr.bonus;
                        env.credit_exploration(fr.bonus);
                    }
                }
            }
        }
        ret += reward;
        steps += 1;
        let info = done.then(|| EpisodeInfo { ret, steps, outcome, max_x: env.max_x(), level: step_level, from_start, start_x, tele: *env.telemetry(), actions: acts });
        if done {
            if let Some(fr) = &frontier {
                if from_path {
                    fr.archive.lock().expect("archive").report_path(step_level, outcome == Outcome::Cleared);
                }
                // A faster win from the level start: replay it to check it is deterministic and keep a state every few steps.
                if outcome == Outcome::Cleared && from_start && fr.path_share > 0.0 {
                    let known = fr.archive.lock().expect("archive").best_path_steps(step_level);
                    if known.is_none_or(|k| (trace.len() as u32) * 100 < k * 97) {
                        record_win(&mut env, step_level, &trace, fr);
                    }
                }
            }
            trace.clear();
            let next = sampler.lock().expect("sampler").sample(&mut rng);
            let mut restart = None;
            from_path = false;
            if let Some(fr) = &frontier {
                let u = crate::rng::unit_f32(&mut rng);
                if u < fr.path_share {
                    restart = fr.archive.lock().expect("archive").sample_path(next, &mut rng);
                    from_path = restart.is_some();
                }
                if restart.is_none() && u >= fr.path_share && u < fr.path_share + fr.start_share {
                    restart = fr.archive.lock().expect("archive").sample(next, &mut rng);
                }
            }
            from_start = restart.is_none();
            let _ = env.reset_from(next, restart.as_ref().map(|(_, s)| s.as_slice()));
            ret = 0.0;
            steps = 0;
            start_x = env.x();
            acts = [0u16; ACTIONS.len()];
        }
        if replies.send(Reply { id, obs: env.observation().to_vec(), ent: env.entities().to_vec(), reward, done, info, level: env.level(), step_level }).is_err() {
            break;
        }
    }
}

impl VecEnv {
    pub fn new(n: usize, core: &Path, rom: &Path, levels: Arc<LevelSet>, sampler: SharedSampler, k_speed: f32, seed: u64) -> Result<Self, String> {
        Self::with_archive(n, core, rom, levels, sampler, None, k_speed, seed)
    }

    /// `frontier` = shared archive of discovered states, the share of episodes that start from one, and the discovery bonus.
    #[allow(clippy::too_many_arguments)]
    pub fn with_archive(n: usize, core: &Path, rom: &Path, levels: Arc<LevelSet>, sampler: SharedSampler, frontier: Option<Frontier>, k_speed: f32, seed: u64) -> Result<Self, String> {
        let (reply_tx, replies) = channel();
        let (mut commands, mut workers) = (Vec::new(), Vec::new());
        for id in 0..n {
            let copy = std::env::temp_dir().join(format!("mario_core_{}_{id}.so", std::process::id()));
            std::fs::copy(core, &copy).map_err(|e| format!("copy core: {e}"))?;
            let (tx, rx) = channel();
            commands.push(tx);
            let (rom, levels, sampler, frontier, reply_tx) = (rom.to_path_buf(), levels.clone(), sampler.clone(), frontier.clone(), reply_tx.clone());
            let worker_seed = seed.wrapping_add(id as u64 + 1);
            workers.push(std::thread::spawn(move || worker(id, copy, rom, levels, sampler, frontier, k_speed, worker_seed, rx, reply_tx)));
        }
        drop(reply_tx);
        let obs_len = obs::len();
        let mut obs = vec![0u8; n * obs_len];
        let mut ent = vec![0.0f32; n * ENT_LEN];
        let mut level_of = vec![0usize; n];
        for _ in 0..n {
            let r = replies.recv().map_err(|_| "a worker failed to start".to_string())?;
            obs[r.id * obs_len..(r.id + 1) * obs_len].copy_from_slice(&r.obs);
            ent[r.id * ENT_LEN..(r.id + 1) * ENT_LEN].copy_from_slice(&r.ent);
            level_of[r.id] = r.level;
        }
        Ok(Self { commands, replies, workers, obs, ent, levels: level_of, n })
    }

    pub fn step(&mut self, actions: &[usize]) -> StepBatch {
        for (tx, &a) in self.commands.iter().zip(actions) {
            tx.send(a).expect("worker died");
        }
        let mut batch = StepBatch { rewards: vec![0.0; self.n], dones: vec![false; self.n], step_levels: vec![0; self.n], finished: Vec::new() };
        let obs_len = obs::len();
        for _ in 0..self.n {
            let r = self.replies.recv().expect("worker died");
            self.obs[r.id * obs_len..(r.id + 1) * obs_len].copy_from_slice(&r.obs);
            self.ent[r.id * ENT_LEN..(r.id + 1) * ENT_LEN].copy_from_slice(&r.ent);
            self.levels[r.id] = r.level;
            batch.rewards[r.id] = r.reward;
            batch.dones[r.id] = r.done;
            batch.step_levels[r.id] = r.step_level;
            batch.finished.extend(r.info);
        }
        batch
    }
}

impl Drop for VecEnv {
    fn drop(&mut self) {
        self.commands.clear(); // closes the channels, which ends the workers
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}
