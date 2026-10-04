//! N environments, one worker thread each. Every worker loads its own copy of the core `.so`
//! because the core keeps global state, so a shared handle would be one emulator.

use crate::env::{MarioEnv, OBS_LEN, Outcome, StepResult};
use std::path::Path;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread::JoinHandle;

#[derive(Clone, Copy, Debug)]
pub struct EpisodeInfo {
    pub ret: f32,
    pub steps: u32,
    pub outcome: Outcome,
    pub max_x: u16,
    /// Which start state the episode began from (0 = real level start).
    pub start: usize,
}

struct Reply {
    id: usize,
    obs: Vec<u8>,
    reward: f32,
    done: bool,
    info: Option<EpisodeInfo>,
}

pub struct VecEnv {
    commands: Vec<Sender<usize>>,
    replies: Receiver<Reply>,
    workers: Vec<JoinHandle<()>>,
    /// Latest observation of every env, `n * OBS_LEN` bytes.
    pub obs: Vec<u8>,
    pub n: usize,
}

pub struct StepBatch {
    pub rewards: Vec<f32>,
    pub dones: Vec<bool>,
    pub finished: Vec<EpisodeInfo>,
}

fn worker(id: usize, core: std::path::PathBuf, rom: std::path::PathBuf, starts: Vec<Vec<u8>>, p_start: f32, commands: Receiver<usize>, replies: Sender<Reply>) {
    let mut env = match MarioEnv::with_starts(&core, &rom, starts, p_start, id as u64 + 1) {
        Ok(env) => env,
        Err(e) => {
            eprintln!("worker {id}: {e}");
            return;
        }
    };
    let _ = std::fs::remove_file(&core); // the library stays mapped after unlinking
    let _ = replies.send(Reply { id, obs: env.observation().to_vec(), reward: 0.0, done: false, info: None });
    let (mut ret, mut steps) = (0.0f32, 0u32);
    while let Ok(action) = commands.recv() {
        let StepResult { reward, outcome } = env.step(action);
        ret += reward;
        steps += 1;
        let done = outcome != Outcome::Running;
        let info = done.then(|| EpisodeInfo { ret, steps, outcome, max_x: env.max_x(), start: env.start_index() });
        if done {
            let _ = env.reset();
            ret = 0.0;
            steps = 0;
        }
        if replies.send(Reply { id, obs: env.observation().to_vec(), reward, done, info }).is_err() {
            break;
        }
    }
}

impl VecEnv {
    pub fn new(n: usize, core: &Path, rom: &Path, starts: &[Vec<u8>], p_start: f32) -> Result<Self, String> {
        let (reply_tx, replies) = channel();
        let (mut commands, mut workers) = (Vec::new(), Vec::new());
        for id in 0..n {
            let copy = std::env::temp_dir().join(format!("mario_core_{}_{id}.so", std::process::id()));
            std::fs::copy(core, &copy).map_err(|e| format!("copy core: {e}"))?;
            let (tx, rx) = channel();
            commands.push(tx);
            let (rom, starts, reply_tx) = (rom.to_path_buf(), starts.to_vec(), reply_tx.clone());
            workers.push(std::thread::spawn(move || worker(id, copy, rom, starts, p_start, rx, reply_tx)));
        }
        drop(reply_tx);
        let mut obs = vec![0u8; n * OBS_LEN];
        for _ in 0..n {
            let r = replies.recv().map_err(|_| "a worker failed to start".to_string())?;
            obs[r.id * OBS_LEN..(r.id + 1) * OBS_LEN].copy_from_slice(&r.obs);
        }
        Ok(Self { commands, replies, workers, obs, n })
    }

    pub fn step(&mut self, actions: &[usize]) -> StepBatch {
        for (tx, &a) in self.commands.iter().zip(actions) {
            tx.send(a).expect("worker died");
        }
        let mut batch = StepBatch { rewards: vec![0.0; self.n], dones: vec![false; self.n], finished: Vec::new() };
        for _ in 0..self.n {
            let r = self.replies.recv().expect("worker died");
            self.obs[r.id * OBS_LEN..(r.id + 1) * OBS_LEN].copy_from_slice(&r.obs);
            batch.rewards[r.id] = r.reward;
            batch.dones[r.id] = r.done;
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
