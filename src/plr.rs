//! Prioritized Level Replay over a fixed set of levels. A level's score is the mean |GAE| seen
//! on it (how surprised the value function still is). Sampling mixes a rank-based score weight
//! with a staleness weight, so no level is starved.

use crate::rng::unit_f32;
use std::sync::{Arc, Mutex};

#[derive(Clone, Copy, Debug)]
pub struct PlrConfig {
    /// false = uniform over the levels.
    pub enabled: bool,
    /// Lower = sharper preference for high-score levels.
    pub temperature: f32,
    /// Share of the staleness distribution in the mix.
    pub rho: f32,
    /// EMA rate for score updates.
    pub ema: f32,
}

impl Default for PlrConfig {
    fn default() -> Self {
        Self { enabled: true, temperature: 0.3, rho: 0.1, ema: 0.3 }
    }
}

pub struct LevelSampler {
    cfg: PlrConfig,
    ids: Vec<usize>,
    scores: Vec<f32>,
    seen: Vec<bool>,
    /// Value of `clock` when each level was last sampled.
    last_sampled: Vec<u64>,
    clock: u64,
}

pub type SharedSampler = Arc<Mutex<LevelSampler>>;

impl LevelSampler {
    pub fn new(ids: Vec<usize>, cfg: PlrConfig) -> Self {
        let n = ids.len();
        Self { cfg, ids, scores: vec![0.0; n], seen: vec![false; n], last_sampled: vec![0; n], clock: 0 }
    }

    pub fn shared(ids: Vec<usize>, cfg: PlrConfig) -> SharedSampler {
        Arc::new(Mutex::new(Self::new(ids, cfg)))
    }

    pub fn update(&mut self, level: usize, score: f32) {
        let Some(i) = self.ids.iter().position(|&id| id == level) else { return };
        self.scores[i] = if self.seen[i] { (1.0 - self.cfg.ema) * self.scores[i] + self.cfg.ema * score } else { score };
        self.seen[i] = true;
    }

    /// Record that `level` was just handed out.
    pub fn note_sampled(&mut self, level: usize) {
        if let Some(i) = self.ids.iter().position(|&id| id == level) {
            self.clock += 1;
            self.last_sampled[i] = self.clock;
        }
    }

    pub fn scores(&self) -> Vec<(usize, f32)> {
        self.ids.iter().copied().zip(self.scores.iter().copied()).collect()
    }

    /// Probability of handing out each level.
    pub fn weights(&self) -> Vec<(usize, f32)> {
        let n = self.ids.len();
        if !self.cfg.enabled {
            return self.ids.iter().map(|&id| (id, 1.0 / n as f32)).collect();
        }
        // Rank 1 = highest score; weight (1 / rank)^(1 / temperature). Tied scores share the
        // average weight of the ranks they span, so list order never breaks a tie.
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| self.scores[b].total_cmp(&self.scores[a]));
        let rank_weight = |rank: usize| (1.0 / (rank as f32 + 1.0)).powf(1.0 / self.cfg.temperature);
        let mut p_score = vec![0.0f32; n];
        let mut start = 0;
        while start < n {
            let mut end = start + 1;
            while end < n && self.scores[order[end]] == self.scores[order[start]] {
                end += 1;
            }
            let shared = (start..end).map(rank_weight).sum::<f32>() / (end - start) as f32;
            for &i in &order[start..end] {
                p_score[i] = shared;
            }
            start = end;
        }
        let total: f32 = p_score.iter().sum();
        p_score.iter_mut().for_each(|p| *p /= total);
        // Staleness: how long ago each level was last handed out.
        let mut p_stale: Vec<f32> = self.last_sampled.iter().map(|&t| (self.clock - t) as f32 + 1.0).collect();
        let total: f32 = p_stale.iter().sum();
        p_stale.iter_mut().for_each(|p| *p /= total);
        (0..n).map(|i| (self.ids[i], (1.0 - self.cfg.rho) * p_score[i] + self.cfg.rho * p_stale[i])).collect()
    }

    pub fn sample(&mut self, rng: &mut u64) -> usize {
        let weights = self.weights();
        let mut u = unit_f32(rng);
        let mut chosen = weights[weights.len() - 1].0;
        for &(id, p) in &weights {
            if u < p {
                chosen = id;
                break;
            }
            u -= p;
        }
        self.note_sampled(chosen);
        chosen
    }
}

/// Mean |advantage| per level over one rollout. `None` for levels that did not appear.
pub fn level_scores(adv: &[f32], level_ids: &[usize], num_levels: usize) -> Vec<Option<f32>> {
    let (mut sum, mut count) = (vec![0.0f32; num_levels], vec![0u32; num_levels]);
    for (a, &l) in adv.iter().zip(level_ids) {
        sum[l] += a.abs();
        count[l] += 1;
    }
    (0..num_levels).map(|l| (count[l] > 0).then(|| sum[l] / count[l] as f32)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(enabled: bool) -> PlrConfig {
        PlrConfig { enabled, temperature: 0.3, rho: 0.1, ema: 0.5 }
    }

    #[test]
    fn disabled_is_uniform() {
        let s = LevelSampler::new(vec![3, 7, 9], cfg(false));
        for (_, w) in s.weights() {
            assert!((w - 1.0 / 3.0).abs() < 1e-6);
        }
    }

    #[test]
    fn weights_sum_to_one_and_favor_high_scores() {
        let mut s = LevelSampler::new(vec![0, 1, 2, 3], cfg(true));
        s.update(0, 0.1);
        s.update(1, 5.0);
        s.update(2, 1.0);
        s.update(3, 0.5);
        let w = s.weights();
        let total: f32 = w.iter().map(|(_, p)| p).sum();
        assert!((total - 1.0).abs() < 1e-5);
        let p = |id: usize| w.iter().find(|(i, _)| *i == id).unwrap().1;
        assert!(p(1) > p(2) && p(2) > p(3) && p(3) > p(0));
    }

    #[test]
    fn staleness_lifts_a_level_that_has_not_been_sampled() {
        let mut s = LevelSampler::new(vec![0, 1], PlrConfig { enabled: true, temperature: 0.3, rho: 0.9, ema: 0.5 });
        s.update(0, 1.0);
        s.update(1, 1.0);
        for _ in 0..50 {
            s.note_sampled(0);
        }
        let w = s.weights();
        let p = |id: usize| w.iter().find(|(i, _)| *i == id).unwrap().1;
        assert!(p(1) > p(0), "level 1 is staler");
    }

    #[test]
    fn tied_scores_get_equal_weight() {
        // Before any update every score is 0; list order must not decide the weights.
        let s = LevelSampler::new(vec![0, 1, 2, 3], cfg(true));
        let w = s.weights();
        for (_, p) in &w {
            assert!((p - 0.25).abs() < 1e-6, "got {w:?}");
        }
        let mut s = LevelSampler::new(vec![0, 1, 2], cfg(true));
        s.update(2, 3.0); // one clear leader, two tied behind it
        let w = s.weights();
        let p = |id: usize| w.iter().find(|(i, _)| *i == id).unwrap().1;
        assert!(p(2) > p(0));
        assert!((p(0) - p(1)).abs() < 1e-6);
    }

    #[test]
    fn sample_only_returns_known_ids_and_covers_all() {
        let mut s = LevelSampler::new(vec![4, 8, 15], cfg(true));
        let mut rng = 12345u64;
        let mut seen = std::collections::HashSet::new();
        for _ in 0..500 {
            let id = s.sample(&mut rng);
            assert!([4, 8, 15].contains(&id));
            seen.insert(id);
        }
        assert_eq!(seen.len(), 3);
    }

    #[test]
    fn ema_blends_old_and_new() {
        let mut s = LevelSampler::new(vec![0], cfg(true));
        s.update(0, 2.0);
        assert_eq!(s.scores()[0].1, 2.0, "first update sets the score");
        s.update(0, 4.0);
        assert_eq!(s.scores()[0].1, 3.0, "ema 0.5 of 2 and 4");
    }

    #[test]
    fn level_scores_average_abs_advantage_per_level() {
        let adv = [1.0, -3.0, 2.0, 0.5];
        let ids = [0, 0, 2, 2];
        let s = level_scores(&adv, &ids, 3);
        assert_eq!(s[0], Some(2.0));
        assert_eq!(s[1], None);
        assert_eq!(s[2], Some(1.25));
    }
}
