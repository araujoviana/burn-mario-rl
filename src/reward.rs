//! Reward terms that do not need the emulator, so they can be tested directly.

use crate::env::FRAME_SKIP;
use crate::ram::TIMER_TICK_FRAMES;

pub const REWARD_CLEAR: f32 = 100.0;
const MIN_STEP_CAP: u32 = 600;

/// Bonus for clearing the level: `100 * (1 + k * fraction of the timer left)`.
pub fn clear_bonus(timer_left: u32, timer_start: u32, k: f32) -> f32 {
    let frac = if timer_start == 0 { 0.0 } else { (timer_left.min(timer_start) as f32) / timer_start as f32 };
    REWARD_CLEAR * (1.0 + k * frac)
}

/// Agent steps until the in-game timer would reach zero, with a floor for short timers.
pub fn step_cap(timer_start: u32) -> u32 {
    (timer_start * TIMER_TICK_FRAMES / FRAME_SKIP).max(MIN_STEP_CAP)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bonus_scales_with_time_left() {
        assert_eq!(clear_bonus(0, 300, 1.0), 100.0);
        assert_eq!(clear_bonus(300, 300, 1.0), 200.0);
        assert_eq!(clear_bonus(150, 300, 1.0), 150.0);
        assert_eq!(clear_bonus(150, 300, 0.0), 100.0);
    }

    #[test]
    fn bonus_handles_zero_start_timer() {
        assert_eq!(clear_bonus(0, 0, 1.0), 100.0);
    }

    #[test]
    fn cap_follows_timer() {
        // timer ticks every TIMER_TICK_FRAMES frames; one agent step is FRAME_SKIP frames.
        let cap = step_cap(300);
        assert_eq!(cap, 300 * crate::ram::TIMER_TICK_FRAMES / crate::env::FRAME_SKIP);
        assert!(step_cap(0) >= 600, "never below the floor");
    }
}
