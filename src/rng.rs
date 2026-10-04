//! Tiny xorshift generator shared by the sampler and the augmentation.

pub fn next_u64(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

/// Uniform in [0, 1).
pub fn unit_f32(state: &mut u64) -> f32 {
    (next_u64(state) >> 40) as f32 / (1u64 << 24) as f32
}
