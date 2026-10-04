//! The set of levels the agent trains and is tested on, stored as `manifest.txt` plus one
//! `<id>.state` file per level.

use std::path::Path;

pub struct Level {
    pub id: usize,
    /// `$13BF` after loading; informational (names the level in logs).
    pub translevel: u8,
    /// Overworld cursor position the level was entered from.
    pub map_x: u16,
    pub map_y: u16,
    pub held_out: bool,
    pub state: Vec<u8>,
}

pub struct LevelSet {
    pub levels: Vec<Level>,
}

/// Every fifth level is held out (index 4, 9, ...), at most four, none when there are fewer than five.
pub fn assign_split(count: usize) -> Vec<bool> {
    let mut held = 0;
    (0..count)
        .map(|i| {
            let hold = i % 5 == 4 && held < 4;
            held += hold as usize;
            hold
        })
        .collect()
}

impl LevelSet {
    pub fn train_ids(&self) -> Vec<usize> {
        self.levels.iter().filter(|l| !l.held_out).map(|l| l.id).collect()
    }

    pub fn held_out_ids(&self) -> Vec<usize> {
        self.levels.iter().filter(|l| l.held_out).map(|l| l.id).collect()
    }

    pub fn save(&self, dir: &Path) -> Result<(), String> {
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        let mut manifest = String::new();
        for l in &self.levels {
            manifest.push_str(&format!("{} {} {} {} {}\n", l.id, l.translevel, l.map_x, l.map_y, if l.held_out { "held_out" } else { "train" }));
            std::fs::write(dir.join(format!("{}.state", l.id)), &l.state).map_err(|e| e.to_string())?;
        }
        std::fs::write(dir.join("manifest.txt"), manifest).map_err(|e| e.to_string())
    }

    pub fn load(dir: &Path) -> Result<Self, String> {
        let manifest = std::fs::read_to_string(dir.join("manifest.txt")).map_err(|e| format!("{}: {e} (run make_levels)", dir.display()))?;
        let mut levels = Vec::new();
        for line in manifest.lines().filter(|l| !l.trim().is_empty()) {
            let f: Vec<&str> = line.split_whitespace().collect();
            if f.len() != 5 {
                return Err(format!("bad manifest line: {line}"));
            }
            let id: usize = f[0].parse().map_err(|_| format!("bad id in {line}"))?;
            let translevel: u8 = f[1].parse().map_err(|_| format!("bad translevel in {line}"))?;
            let map_x: u16 = f[2].parse().map_err(|_| format!("bad map_x in {line}"))?;
            let map_y: u16 = f[3].parse().map_err(|_| format!("bad map_y in {line}"))?;
            let state = std::fs::read(dir.join(format!("{id}.state"))).map_err(|e| e.to_string())?;
            levels.push(Level { id, translevel, map_x, map_y, held_out: f[4] == "held_out", state });
        }
        Ok(Self { levels })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_holds_out_every_fifth_capped_at_four() {
        assert_eq!(assign_split(3), vec![false, false, false]);
        let s = assign_split(15);
        assert_eq!(s.iter().filter(|&&h| h).count(), 3);
        assert!(s[4] && s[9] && s[14]);
        let big = assign_split(40);
        assert_eq!(big.iter().filter(|&&h| h).count(), 4);
    }

    #[test]
    fn manifest_round_trip() {
        let dir = std::env::temp_dir().join(format!("levels_test_{}", std::process::id()));
        let set = LevelSet {
            levels: vec![
                Level { id: 0, translevel: 5, map_x: 152, map_y: 136, held_out: false, state: vec![1, 2, 3] },
                Level { id: 1, translevel: 9, map_x: 264, map_y: 24, held_out: true, state: vec![4, 5] },
            ],
        };
        set.save(&dir).unwrap();
        let back = LevelSet::load(&dir).unwrap();
        assert_eq!(back.levels.len(), 2);
        assert_eq!(back.levels[1].translevel, 9);
        assert_eq!((back.levels[1].map_x, back.levels[1].map_y), (264, 24));
        assert!(back.levels[1].held_out);
        assert_eq!(back.levels[1].state, vec![4, 5]);
        assert_eq!(back.train_ids(), vec![0]);
        assert_eq!(back.held_out_ids(), vec![1]);
        let _ = std::fs::remove_dir_all(dir);
    }
}
