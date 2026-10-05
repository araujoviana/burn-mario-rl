//! Go-Explore style archive: save states taken the first time Mario stands on solid ground in each
//! cell of a level (x and y), so new episodes can start from places the agent has already discovered
//! and explore on from there. Cells, not just distance, so routes that go up or back left are kept.
//! Shared by every worker. Held-out evaluation never uses it.

use crate::rng::unit_f32;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Size of one cell in pixels.
pub const CELL_X: u16 = 96;
pub const CELL_Y: u16 = 64;
/// Memory bound: a state is about 0.8 MB.
const MAX_CELLS_PER_LEVEL: usize = 160;

struct Cell {
    x: u16,
    state: Vec<u8>,
    /// Episodes started from this cell so far.
    visits: u32,
}

/// States along the shortest known winning run of a level, in play order. Episodes start from the end of
/// the path first (`frac` of it), and the start moves back as the agent keeps clearing from there.
#[derive(Default)]
struct WinPath {
    steps: u32,
    states: Vec<Vec<u8>>,
    /// Share of the path, counted from its end, that starts are drawn from.
    frac: f32,
    wins: u32,
    fails: u32,
}

/// Agent steps between two states stored along a winning path.
pub const PATH_STRIDE: usize = 8;
const PATH_FRAC_START: f32 = 0.1;
/// Path episodes per curriculum decision.
const PATH_WINDOW: u32 = 20;

#[derive(Default)]
pub struct Archive {
    levels: Vec<HashMap<(u16, u16), Cell>>,
    paths: Vec<WinPath>,
}

pub type SharedArchive = Arc<Mutex<Archive>>;

/// Archive plus the knobs that use it.
#[derive(Clone)]
pub struct Frontier {
    pub archive: SharedArchive,
    /// Share of episodes that start from an archived state instead of the level start.
    pub start_share: f32,
    /// Reward for discovering a new cell (first visit by any worker).
    pub bonus: f32,
    /// Share of episodes that start along a recorded winning path (backward curriculum).
    pub path_share: f32,
}

fn cell_of(x: u16, y: u16) -> (u16, u16) {
    (x / CELL_X, y / CELL_Y)
}

impl Archive {
    pub fn shared(levels: usize) -> SharedArchive {
        Arc::new(Mutex::new(Self { levels: (0..levels).map(|_| HashMap::new()).collect(), paths: (0..levels).map(|_| WinPath::default()).collect() }))
    }

    /// True when the cell containing (`x`, `y`) has no state yet and the level still has room.
    pub fn wants(&self, level: usize, x: u16, y: u16) -> bool {
        let cells = &self.levels[level];
        cells.len() < MAX_CELLS_PER_LEVEL && !cells.contains_key(&cell_of(x, y))
    }

    /// Stores `state` if the cell is new; true when it was.
    pub fn offer(&mut self, level: usize, x: u16, y: u16, state: Vec<u8>) -> bool {
        if !self.wants(level, x, y) {
            return false;
        }
        self.levels[level].insert(cell_of(x, y), Cell { x, state, visits: 0 });
        true
    }

    /// A stored state for `level` and its x. Half the time one of the two cells furthest to the right,
    /// otherwise a cell chosen with weight 1/sqrt(1 + visits), so rarely-tried cells are favoured.
    pub fn sample(&mut self, level: usize, rng: &mut u64) -> Option<(u16, Vec<u8>)> {
        let cells = &mut self.levels[level];
        if cells.is_empty() {
            return None;
        }
        let mut keys: Vec<(u16, u16)> = cells.keys().copied().collect();
        keys.sort_by_key(|k| std::cmp::Reverse(cells[k].x)); // furthest right first; ties are fine
        let pick = if unit_f32(rng) < 0.5 {
            keys[(unit_f32(rng) * keys.len().min(2) as f32) as usize]
        } else {
            let weights: Vec<f32> = keys.iter().map(|k| 1.0 / (1.0 + cells[k].visits as f32).sqrt()).collect();
            let mut u = unit_f32(rng) * weights.iter().sum::<f32>();
            let mut chosen = keys[keys.len() - 1];
            for (k, w) in keys.iter().zip(&weights) {
                if u < *w {
                    chosen = *k;
                    break;
                }
                u -= w;
            }
            chosen
        };
        let cell = cells.get_mut(&pick)?;
        cell.visits += 1;
        Some((cell.x, cell.state.clone()))
    }

    /// Steps of the shortest winning path stored for `level`, if any.
    pub fn best_path_steps(&self, level: usize) -> Option<u32> {
        (!self.paths[level].states.is_empty()).then_some(self.paths[level].steps)
    }

    /// Stores a verified winning path (a state every `PATH_STRIDE` steps) if it beats the stored one.
    pub fn offer_path(&mut self, level: usize, steps: u32, states: Vec<Vec<u8>>) {
        let p = &mut self.paths[level];
        if states.is_empty() || (!p.states.is_empty() && steps >= p.steps) {
            return;
        }
        *p = WinPath { steps, states, frac: if p.frac > 0.0 { p.frac } else { PATH_FRAC_START }, wins: 0, fails: 0 };
    }

    /// A state from the last `frac` of the winning path, and its index (to tell when the start has reached the beginning).
    pub fn sample_path(&self, level: usize, rng: &mut u64) -> Option<(u16, Vec<u8>)> {
        let p = &self.paths[level];
        let n = p.states.len();
        if n == 0 {
            return None;
        }
        let first = ((n as f32) * (1.0 - p.frac)).floor() as usize;
        let i = (first + (unit_f32(rng) * (n - first) as f32) as usize).min(n - 1);
        Some((0, p.states[i].clone()))
    }

    /// Result of an episode that started on the path of `level`; moves the start back (or forward) as it goes.
    pub fn report_path(&mut self, level: usize, cleared: bool) {
        let p = &mut self.paths[level];
        if cleared { p.wins += 1 } else { p.fails += 1 }
        if p.wins + p.fails >= PATH_WINDOW {
            let rate = p.wins as f32 / (p.wins + p.fails) as f32;
            if rate > 0.7 {
                p.frac = (p.frac + 0.1).min(1.0);
            } else if rate < 0.3 {
                p.frac = (p.frac - 0.05).max(PATH_FRAC_START / 2.0);
            }
            p.wins = 0;
            p.fails = 0;
        }
    }

    /// (steps, share of the path starts are drawn from) per level with a stored path, for logging.
    pub fn path_status(&self) -> Vec<(usize, u32, f32)> {
        self.paths.iter().enumerate().filter(|(_, p)| !p.states.is_empty()).map(|(l, p)| (l, p.steps, p.frac)).collect()
    }

    /// Filled cells per level, for logging.
    pub fn counts(&self) -> Vec<usize> {
        self.levels.iter().map(|c| c.len()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn archive() -> Archive {
        Archive { levels: vec![HashMap::new()], paths: vec![WinPath::default()] }
    }

    #[test]
    fn keeps_first_state_per_cell() {
        let mut a = archive();
        assert!(a.wants(0, 400, 300));
        assert!(a.offer(0, 400, 300, vec![7]));
        assert!(!a.offer(0, 410, 310, vec![9]), "same cell");
        assert!(a.offer(0, 400, 100, vec![8]), "same x, higher cell");
        assert_eq!(a.counts()[0], 2);
    }

    #[test]
    fn samples_stored_states_and_counts_visits() {
        let mut a = archive();
        a.offer(0, 400, 300, vec![7]);
        let mut rng = 5u64;
        let (x, state) = a.sample(0, &mut rng).unwrap();
        assert_eq!((x, state), (400, vec![7]));
        assert_eq!(a.levels[0][&cell_of(400, 300)].visits, 1);
    }

    #[test]
    fn rarely_visited_cells_are_favoured() {
        let mut a = archive();
        for i in 0..5u16 {
            a.offer(0, 100 + i * CELL_X, 300, vec![i as u8]);
        }
        a.levels[0].get_mut(&cell_of(100, 300)).unwrap().visits = 10_000;
        let mut rng = 3u64;
        let hits = (0..400).filter(|_| a.sample(0, &mut rng).unwrap().1 == vec![0]).count();
        assert!(hits < 40, "the heavily visited cell is rarely picked, got {hits}/400");
    }

    #[test]
    fn path_start_moves_back_as_the_agent_clears() {
        let mut a = archive();
        a.offer_path(0, 200, (0..50u8).map(|i| vec![i]).collect());
        assert_eq!(a.best_path_steps(0), Some(200));
        let mut rng = 9u64;
        // at first only the last tenth of the path (states 45..50) is used
        assert!((0..50).all(|_| a.sample_path(0, &mut rng).unwrap().1[0] >= 45));
        for _ in 0..PATH_WINDOW {
            a.report_path(0, true);
        }
        assert!((0..200).any(|_| a.sample_path(0, &mut rng).unwrap().1[0] < 45), "start moved back after a run of clears");
        // a worse path does not replace the best one
        a.offer_path(0, 300, vec![vec![99]]);
        assert_eq!(a.best_path_steps(0), Some(200));
        a.offer_path(0, 150, vec![vec![7]]);
        assert_eq!(a.best_path_steps(0), Some(150));
    }

    #[test]
    fn empty_level_samples_nothing() {
        assert!(archive().sample(0, &mut 3u64).is_none());
    }
}
