//! Super Mario World work-RAM addresses (offsets from $7E0000) and decoding helpers.
//! Every address here was checked with `examples/warp_probe.rs`; see the comments for how.

use crate::emulator::Emulator;
use crate::env::game_mode;

pub const GAME_MODE: usize = 0x100;
pub const PLAYER_STATE: usize = 0x71;
pub const PLAYER_X: usize = 0x94; // 16 bit
pub const PLAYER_Y: usize = 0x96; // 16 bit
pub const PLAYER_SPEED_X: usize = 0x7B; // signed
pub const PLAYER_SPEED_Y: usize = 0x7D; // signed
pub const PLAYER_BLOCKED: usize = 0x77; // bit 2 set = standing on something
pub const POWERUP: usize = 0x19;
pub const TRANSLEVEL: usize = 0x13BF;
pub const MAP_X: usize = 0x1F17; // overworld cursor, 16 bit, pixels
pub const MAP_Y: usize = 0x1F19; // overworld cursor, 16 bit, pixels
pub const END_LEVEL_TIMER: usize = 0x1493;
pub const LAYOUT_FLAGS: usize = 0x5B; // bit 0 set = vertical level
pub const TIMER_HUNDREDS: usize = 0xF31;
pub const TIMER_TENS: usize = 0xF32;
pub const TIMER_ONES: usize = 0xF33;

pub const SPRITE_SLOTS: usize = 12;
pub const SPRITE_STATUS: usize = 0x14C8; // 0 = empty slot
pub const SPRITE_TYPE: usize = 0x9E;
pub const SPRITE_X_LO: usize = 0xE4;
pub const SPRITE_X_HI: usize = 0x14E0;
pub const SPRITE_Y_LO: usize = 0xD8;
pub const SPRITE_Y_HI: usize = 0x14D4;

/// Emulator frames per one tick of the in-game timer (measured by `examples/warp_probe.rs`).
pub const TIMER_TICK_FRAMES: u32 = 41;

pub fn u16_at(ram: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([ram[offset], ram[offset + 1]])
}

/// In-game countdown timer, 0..=999.
pub fn timer(ram: &[u8]) -> u32 {
    ram[TIMER_HUNDREDS] as u32 * 100 + ram[TIMER_TENS] as u32 * 10 + ram[TIMER_ONES] as u32
}

pub fn is_vertical(ram: &[u8]) -> bool {
    ram[LAYOUT_FLAGS] & 1 != 0
}

/// Load `overworld`, teleport the map cursor to pixel (`x`, `y`), press A, and return a save state
/// on the first controllable frame of the level there. Errors when the tile is not a level.
///
/// Writing the translevel (`$13BF`) or level number does nothing: the game derives both from the
/// cursor position in the frame A is pressed. Teleporting the cursor is what works. Positions
/// that are not level tiles either never leave the map or load a blank "level" with timer 0.
pub fn warp_to_position(emu: &mut Emulator, overworld: &[u8], x: u16, y: u16) -> Result<Vec<u8>, String> {
    emu.load_state(overworld)?;
    emu.write_ram(MAP_X, &x.to_le_bytes());
    emu.write_ram(MAP_Y, &y.to_le_bytes());
    let mut entered = false;
    for f in 0..400u32 {
        emu.set_buttons(if f > 4 && f % 60 < 5 { crate::emulator::button::A } else { 0 });
        emu.run_frame();
        if game_mode(emu) == 0x14 {
            entered = true;
            break;
        }
    }
    emu.set_buttons(0);
    if !entered {
        return Err(format!("({x},{y}): no level entered"));
    }
    for _ in 0..60 {
        emu.run_frame(); // fade in, Mario lands
    }
    if timer(emu.ram()) == 0 {
        return Err(format!("({x},{y}): blank level (timer 0)"));
    }
    emu.save_state()
}
