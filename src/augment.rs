//! Random-shift augmentation on observation bytes: pad by edge replication, crop back, which is
//! the same as sampling each output pixel from the clamped shifted coordinate.

use crate::env::{OBS_H, OBS_LEN, OBS_W, STACK};
use crate::rng::next_u64;

/// Output (x, y) shows source (x + dx, y + dy), clamped to the image.
pub fn shift_plane(src: &[u8], dst: &mut [u8], dx: i32, dy: i32) {
    for y in 0..OBS_H {
        let sy = (y as i32 + dy).clamp(0, OBS_H as i32 - 1) as usize;
        for x in 0..OBS_W {
            let sx = (x as i32 + dx).clamp(0, OBS_W as i32 - 1) as usize;
            dst[y * OBS_W + x] = src[sy * OBS_W + sx];
        }
    }
}

/// One shift in `[-pad, pad]` per axis for each sample, shared by all its stacked planes.
pub fn random_shift(src: &[u8], out: &mut [u8], samples: usize, pad: usize, rng: &mut u64) {
    let plane = OBS_H * OBS_W;
    let span = 2 * pad as u64 + 1;
    for s in 0..samples {
        let dx = (next_u64(rng) % span) as i32 - pad as i32;
        let dy = (next_u64(rng) % span) as i32 - pad as i32;
        for p in 0..STACK {
            let o = s * OBS_LEN + p * plane;
            shift_plane(&src[o..o + plane], &mut out[o..o + plane], dx, dy);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp() -> Vec<u8> {
        (0..OBS_LEN).map(|i| (i % 251) as u8).collect()
    }

    #[test]
    fn zero_pad_is_identity() {
        let src = ramp();
        let mut out = vec![0u8; OBS_LEN];
        let mut rng = 1u64;
        random_shift(&src, &mut out, 1, 0, &mut rng);
        assert_eq!(src, out);
    }

    #[test]
    fn shift_plane_moves_pixels_and_replicates_edges() {
        let mut src = vec![0u8; OBS_H * OBS_W];
        for y in 0..OBS_H {
            for x in 0..OBS_W {
                src[y * OBS_W + x] = x as u8;
            }
        }
        let mut dst = vec![0u8; OBS_H * OBS_W];
        shift_plane(&src, &mut dst, 2, 0);
        assert_eq!(dst[10], 12, "pixel at x=10 now shows what was at x=12");
        assert_eq!(dst[OBS_W - 1], (OBS_W - 1) as u8, "right edge is replicated");
    }

    #[test]
    fn same_shift_for_every_plane_of_a_sample() {
        // Identical planes must stay identical after a shared shift.
        let plane = OBS_H * OBS_W;
        let one: Vec<u8> = (0..plane).map(|i| (i * 7 % 253) as u8).collect();
        let src: Vec<u8> = (0..STACK).flat_map(|_| one.clone()).collect();
        let mut out = vec![0u8; OBS_LEN];
        let mut rng = 42u64;
        random_shift(&src, &mut out, 1, 4, &mut rng);
        for p in 1..STACK {
            assert_eq!(out[..plane], out[p * plane..(p + 1) * plane]);
        }
    }
}
