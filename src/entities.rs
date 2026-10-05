//! Entity observation: the 12 sprite slots plus Mario's motion, read straight from RAM.

use crate::ram::*;

pub const ENT_LEN: usize = SPRITE_SLOTS * 5 + 10;
const CAMERA_X: usize = 0x1A;
const CAMERA_Y: usize = 0x1C;

fn clamp_unit(v: i32, scale: f32) -> f32 {
    (v as f32 / scale).clamp(-1.0, 1.0)
}

pub fn entity_vector(ram: &[u8], out: &mut [f32]) {
    debug_assert_eq!(out.len(), ENT_LEN);
    let (mx, my) = (u16_at(ram, PLAYER_X) as i32, u16_at(ram, PLAYER_Y) as i32);
    for i in 0..SPRITE_SLOTS {
        let slot = &mut out[i * 5..(i + 1) * 5];
        let status = ram[SPRITE_STATUS + i];
        if status == 0 {
            slot.fill(0.0);
            continue;
        }
        let sx = ram[SPRITE_X_LO + i] as i32 | (ram[SPRITE_X_HI + i] as i32) << 8;
        let sy = ram[SPRITE_Y_LO + i] as i32 | (ram[SPRITE_Y_HI + i] as i32) << 8;
        slot[0] = 1.0;
        slot[1] = ram[SPRITE_TYPE + i] as f32 / 255.0;
        slot[2] = clamp_unit(sx - mx, 128.0);
        slot[3] = clamp_unit(sy - my, 128.0);
        slot[4] = status as f32 / 15.0;
    }
    let m = &mut out[SPRITE_SLOTS * 5..];
    m[0] = clamp_unit(ram[PLAYER_SPEED_X] as i8 as i32, 64.0);
    m[1] = clamp_unit(ram[PLAYER_SPEED_Y] as i8 as i32, 64.0);
    m[2] = (ram[PLAYER_BLOCKED] & 4 != 0) as u8 as f32;
    m[3] = ram[POWERUP] as f32 / 3.0;
    m[4] = clamp_unit(mx - u16_at(ram, CAMERA_X) as i32, 256.0);
    m[5] = clamp_unit(my - u16_at(ram, CAMERA_Y) as i32, 256.0);
    // switch timers and water: what a blue P-switch run or an on/off block room depends on
    m[6] = ram[BLUE_SWITCH_TIMER] as f32 / 255.0;
    m[7] = ram[SILVER_SWITCH_TIMER] as f32 / 255.0;
    m[8] = (ram[ONOFF_SWITCH] & 1) as f32;
    m[9] = (ram[SWIMMING] != 0) as u8 as f32;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn blank() -> Vec<u8> {
        vec![0u8; 0x20000]
    }

    #[test]
    fn empty_slots_are_zero() {
        let ram = blank();
        let mut out = vec![9.0; ENT_LEN];
        entity_vector(&ram, &mut out);
        assert!(out[..SPRITE_SLOTS * 5].iter().all(|&v| v == 0.0));
    }

    #[test]
    fn sprite_is_encoded_relative_to_mario() {
        let mut ram = blank();
        ram[PLAYER_X..PLAYER_X + 2].copy_from_slice(&100u16.to_le_bytes());
        ram[PLAYER_Y..PLAYER_Y + 2].copy_from_slice(&200u16.to_le_bytes());
        ram[SPRITE_STATUS + 2] = 8;
        ram[SPRITE_TYPE + 2] = 51;
        ram[SPRITE_X_LO + 2] = 164; // 64 pixels right of Mario
        ram[SPRITE_Y_LO + 2] = 180; // 20 pixels above
        let mut out = vec![0.0; ENT_LEN];
        entity_vector(&ram, &mut out);
        let s = &out[2 * 5..3 * 5];
        assert_eq!(s[0], 1.0);
        assert!((s[1] - 51.0 / 255.0).abs() < 1e-6);
        assert!((s[2] - 0.5).abs() < 1e-6);
        assert!((s[3] + 20.0 / 128.0).abs() < 1e-6);
        assert!((s[4] - 8.0 / 15.0).abs() < 1e-6);
    }

    #[test]
    fn far_sprites_are_clamped() {
        let mut ram = blank();
        ram[SPRITE_STATUS] = 8;
        ram[SPRITE_X_LO] = 0;
        ram[SPRITE_X_HI] = 5; // x = 1280
        let mut out = vec![0.0; ENT_LEN];
        entity_vector(&ram, &mut out);
        assert_eq!(out[2], 1.0);
    }

    #[test]
    fn mario_speed_is_signed() {
        let mut ram = blank();
        ram[PLAYER_SPEED_X] = (-32i8) as u8;
        ram[PLAYER_BLOCKED] = 4;
        let mut out = vec![0.0; ENT_LEN];
        entity_vector(&ram, &mut out);
        let m = &out[SPRITE_SLOTS * 5..];
        assert_eq!(m[0], -0.5);
        assert_eq!(m[2], 1.0);
    }
}
