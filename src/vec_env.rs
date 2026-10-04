//! N environments, one worker thread each. Every worker loads its own copy of the core `.so`
//! because the core keeps global state, so a shared handle would be one emulator.
//! Workers pick the level of each new episode from a shared `LevelSampler`.

use crate::entities::ENT_LEN;
use crate::env::{MarioEnv, OBS_LEN, Outcome, StepResult};
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
    /// Latest observation of every env, `n * OBS_LEN` bytes.
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

#[allow(clippy::too_many_arguments)]
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
    let _ = replies.send(Reply { id, obs: env.observation().to_vec(), ent: env.entities().to_vec(), reward: 0.0, done: false, info: None, level: env.level(), step_level: env.level() });
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
        if replies.send(Reply { id, obs: env.observation().to_vec(), ent: env.entities().to_vec(), reward, done, info, level: env.level(), step_level }).is_err() {
            break;
        }
    }
}

impl VecEnv {
    pub fn new(n: usize, core: &Path, rom: &Path, levels: Arc<LevelSet>, sampler: SharedSampler, k_speed: f32, seed: u64) -> Result<Self, String> {
        let (reply_tx, replies) = channel();
        let (mut commands, mut workers) = (Vec::new(), Vec::new());
        for id in 0..n {
            let copy = std::env::temp_dir().join(format!("mario_core_{}_{id}.so", std::process::id()));
            std::fs::copy(core, &copy).map_err(|e| format!("copy core: {e}"))?;
            let (tx, rx) = channel();
            commands.push(tx);
            let (rom, levels, sampler, reply_tx) = (rom.to_path_buf(), levels.clone(), sampler.clone(), reply_tx.clone());
            let worker_seed = seed.wrapping_add(id as u64 + 1);
            workers.push(std::thread::spawn(move || worker(id, copy, rom, levels, sampler, k_speed, worker_seed, rx, reply_tx)));
        }
        drop(reply_tx);
        let mut obs = vec![0u8; n * OBS_LEN];
        let mut ent = vec![0.0f32; n * ENT_LEN];
        let mut level_of = vec![0usize; n];
        for _ in 0..n {
            let r = replies.recv().map_err(|_| "a worker failed to start".to_string())?;
            obs[r.id * OBS_LEN..(r.id + 1) * OBS_LEN].copy_from_slice(&r.obs);
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
        for _ in 0..self.n {
            let r = self.replies.recv().expect("worker died");
            self.obs[r.id * OBS_LEN..(r.id + 1) * OBS_LEN].copy_from_slice(&r.obs);
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
