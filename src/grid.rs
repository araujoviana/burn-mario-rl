//! Tile-grid observation read from work RAM: terrain, coins and sprites in a window around Mario.
//!
//! Layer 1 of a horizontal level is stored as Map16 tile ids, low bytes at `$7EC800` and high bytes at
//! `$7FC800`, in screens of 16x27 tiles (0x1B0 entries). Only horizontal levels are supported; a vertical
//! level yields an all-zero grid. Tile classes (checked against `examples/tile_probe.rs` output):
//! `0x25` is empty, ids `>= 0x100` are solid terrain and blocks, `0x2B` is a coin, anything else is an
//! object (decoration, item block). Sprites with status `>= 8` are drawn into one of two planes.

use crate::ram::*;

pub const GRID_C: usize = 7;
pub const GRID_H: usize = 14;
pub const GRID_W: usize = 20;
pub const GRID_LEN: usize = GRID_C * GRID_H * GRID_W;

const PLANE: usize = GRID_H * GRID_W;
/// Columns of the window behind Mario and rows above his body centre.
const COLS_BEHIND: i32 = 5;
const ROWS_ABOVE: i32 = 8;

const MAP16_LOW: usize = 0xC800;
const MAP16_HIGH: usize = 0x1C800;
const MAP16_SIZE: usize = 0x2000;
const SCREEN_TILES: usize = 0x1B0;
const LEVEL_ROWS: i32 = 27;
const EMPTY: usize = 0x25;
const COIN: usize = 0x2B;

pub const CH_SOLID: usize = 0;
pub const CH_COIN: usize = 1;
pub const CH_OBJECT: usize = 2;
pub const CH_HOSTILE: usize = 3;
pub const CH_FRIENDLY: usize = 4;
/// Sprite planes of the previous call live in `CH_HOSTILE + 2 ..` so the net sees sprite motion.
const PREV_OFFSET: usize = 2;

/// Sprite ids that are not enemies: Yoshi, P-switch, power-ups, key.
pub fn is_friendly(sprite: u8) -> bool {
    matches!(sprite, 0x35 | 0x3E | 0x74..=0x7D | 0x80)
}

fn tile_class(id: usize) -> Option<usize> {
    match id {
        EMPTY => None,
        COIN => Some(CH_COIN),
        i if i >= 0x100 => Some(CH_SOLID),
        _ => Some(CH_OBJECT),
    }
}

/// Fills `out` (`GRID_LEN` bytes, `[GRID_C, GRID_H, GRID_W]`). The sprite planes of the previous call are
/// kept in the last two channels, so `out` must be the same buffer from one step to the next.
pub fn build(ram: &[u8], out: &mut [u8]) {
    debug_assert_eq!(out.len(), GRID_LEN);
    out.copy_within(CH_HOSTILE * PLANE..(CH_FRIENDLY + 1) * PLANE, (CH_HOSTILE + PREV_OFFSET) * PLANE);
    out[..(CH_HOSTILE + PREV_OFFSET) * PLANE].fill(0);
    out[CH_HOSTILE * PLANE..(CH_HOSTILE + PREV_OFFSET) * PLANE].fill(0);
    if is_vertical(ram) {
        return;
    }
    let (mx, my) = (u16_at(ram, PLAYER_X) as i32, u16_at(ram, PLAYER_Y) as i32);
    let (tx, ty) = ((mx + 8) >> 4, (my + 16) >> 4);

    for row in 0..GRID_H as i32 {
        let y = ty - ROWS_ABOVE + row;
        for col in 0..GRID_W as i32 {
            let x = tx - COLS_BEHIND + col;
            let class = if x < 0 {
                Some(CH_SOLID)
            } else if !(0..LEVEL_ROWS).contains(&y) {
                None
            } else {
                let off = (x as usize >> 4) * SCREEN_TILES + y as usize * 16 + (x as usize & 15);
                if off >= MAP16_SIZE {
                    Some(CH_SOLID)
                } else {
                    tile_class((ram[MAP16_HIGH + off] as usize) << 8 | ram[MAP16_LOW + off] as usize)
                }
            };
            if let Some(c) = class {
                out[c * PLANE + row as usize * GRID_W + col as usize] = 255;
            }
        }
    }

    for i in 0..SPRITE_SLOTS {
        if ram[SPRITE_STATUS + i] < 8 {
            continue;
        }
        let sx = ram[SPRITE_X_LO + i] as i32 | (ram[SPRITE_X_HI + i] as i32) << 8;
        let sy = ram[SPRITE_Y_LO + i] as i32 | (ram[SPRITE_Y_HI + i] as i32) << 8;
        let col = ((sx + 8) >> 4) - tx + COLS_BEHIND;
        let row = ((sy + 8) >> 4) - ty + ROWS_ABOVE;
        if (0..GRID_W as i32).contains(&col) && (0..GRID_H as i32).contains(&row) {
            let c = if is_friendly(ram[SPRITE_TYPE + i]) { CH_FRIENDLY } else { CH_HOSTILE };
            out[c * PLANE + row as usize * GRID_W + col as usize] = 255;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn blank_ram() -> Vec<u8> {
        let mut ram = vec![0u8; 0x20000];
        ram[MAP16_LOW..MAP16_LOW + MAP16_SIZE].fill(EMPTY as u8);
        ram
    }

    fn set_tile(ram: &mut [u8], x: usize, y: usize, id: usize) {
        let off = (x >> 4) * SCREEN_TILES + y * 16 + (x & 15);
        ram[MAP16_LOW + off] = id as u8;
        ram[MAP16_HIGH + off] = (id >> 8) as u8;
    }

    fn cell(out: &[u8], c: usize, row: i32, col: i32) -> u8 {
        out[c * PLANE + row as usize * GRID_W + col as usize]
    }

    #[test]
    fn terrain_lands_below_mario() {
        let mut ram = blank_ram();
        ram[PLAYER_X..PLAYER_X + 2].copy_from_slice(&40u16.to_le_bytes()); // tile column 3 (centre 48)
        ram[PLAYER_Y..PLAYER_Y + 2].copy_from_slice(&352u16.to_le_bytes()); // body centre row 23
        for x in 0..40 {
            set_tile(&mut ram, x, 24, 0x100);
        }
        set_tile(&mut ram, 5, 20, COIN);
        let mut out = vec![0u8; GRID_LEN];
        build(&ram, &mut out);
        let mario_col = COLS_BEHIND;
        assert_eq!(cell(&out, CH_SOLID, ROWS_ABOVE + 1, mario_col), 255, "ground one row below Mario's centre");
        assert_eq!(cell(&out, CH_SOLID, ROWS_ABOVE, mario_col), 0);
        // tile column 5 is two columns right of tile column 3 (Mario)
        assert_eq!(cell(&out, CH_COIN, ROWS_ABOVE - 3, mario_col + 2), 255);
    }

    #[test]
    fn left_edge_of_level_is_a_wall() {
        let mut ram = blank_ram();
        ram[PLAYER_X..PLAYER_X + 2].copy_from_slice(&16u16.to_le_bytes());
        ram[PLAYER_Y..PLAYER_Y + 2].copy_from_slice(&352u16.to_le_bytes());
        let mut out = vec![0u8; GRID_LEN];
        build(&ram, &mut out);
        assert_eq!(cell(&out, CH_SOLID, 3, 0), 255);
        assert_eq!(cell(&out, CH_SOLID, 3, GRID_W as i32 - 1), 0);
    }

    #[test]
    fn sprites_keep_previous_plane() {
        let mut ram = blank_ram();
        ram[PLAYER_X..PLAYER_X + 2].copy_from_slice(&160u16.to_le_bytes());
        ram[PLAYER_Y..PLAYER_Y + 2].copy_from_slice(&352u16.to_le_bytes());
        ram[SPRITE_STATUS] = 8;
        ram[SPRITE_TYPE] = 0x04; // a Koopa
        ram[SPRITE_X_LO] = 200;
        ram[SPRITE_Y_LO] = 100;
        ram[SPRITE_Y_HI] = 1; // y = 356
        let mut out = vec![0u8; GRID_LEN];
        build(&ram, &mut out);
        let hostile = out[CH_HOSTILE * PLANE..(CH_HOSTILE + 1) * PLANE].iter().filter(|&&v| v == 255).count();
        assert_eq!(hostile, 1);
        ram[SPRITE_X_LO] = 216; // moved one tile right
        build(&ram, &mut out);
        let now = out[CH_HOSTILE * PLANE..(CH_HOSTILE + 1) * PLANE].iter().position(|&v| v == 255).unwrap();
        let before = out[(CH_HOSTILE + PREV_OFFSET) * PLANE..(CH_HOSTILE + PREV_OFFSET + 1) * PLANE].iter().position(|&v| v == 255).unwrap();
        assert_eq!(now, before + 1);
    }
}
