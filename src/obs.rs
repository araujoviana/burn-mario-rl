//! Which observation the environment produces: stacked grayscale pixels or a tile grid read from RAM.
//! Chosen once per process (`set`) before any environment is created.

use crate::env::{OBS_H, OBS_W, STACK};
use crate::grid::{GRID_C, GRID_H, GRID_W};
use std::sync::OnceLock;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ObsKind {
    Pixels,
    Grid,
}

static KIND: OnceLock<ObsKind> = OnceLock::new();

/// Fixes the observation kind for this process. A second call with a different kind is a bug.
pub fn set(kind: ObsKind) {
    let got = *KIND.get_or_init(|| kind);
    assert_eq!(got, kind, "observation kind already set to {got:?}");
}

pub fn kind() -> ObsKind {
    *KIND.get_or_init(|| ObsKind::Pixels)
}

/// `(channels, height, width)` of one observation.
pub fn dims() -> (usize, usize, usize) {
    match kind() {
        ObsKind::Pixels => (STACK, OBS_H, OBS_W),
        ObsKind::Grid => (GRID_C, GRID_H, GRID_W),
    }
}

/// Bytes in one observation.
pub fn len() -> usize {
    let (c, h, w) = dims();
    c * h * w
}
