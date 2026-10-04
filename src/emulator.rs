//! Minimal libretro host: loads a core `.so`, runs a ROM, exposes frames, input, RAM and save states.

use libloading::Library;
use std::cell::RefCell;
use std::ffi::{CString, c_char, c_uint, c_void};
use std::path::Path;

const ENV_GET_SYSTEM_DIRECTORY: c_uint = 9;
const ENV_SET_PIXEL_FORMAT: c_uint = 10;
const PIXEL_FORMAT_0RGB1555: c_uint = 0;
const PIXEL_FORMAT_XRGB8888: c_uint = 1;
const PIXEL_FORMAT_RGB565: c_uint = 2;
const DEVICE_JOYPAD: c_uint = 1;
const DEVICE_MASK: c_uint = 256;
const MEMORY_SYSTEM_RAM: c_uint = 2;

/// Joypad button ids, also the bit positions in the input mask.
pub mod button {
    pub const B: u16 = 1 << 0;
    pub const Y: u16 = 1 << 1;
    pub const SELECT: u16 = 1 << 2;
    pub const START: u16 = 1 << 3;
    pub const UP: u16 = 1 << 4;
    pub const DOWN: u16 = 1 << 5;
    pub const LEFT: u16 = 1 << 6;
    pub const RIGHT: u16 = 1 << 7;
    pub const A: u16 = 1 << 8;
    pub const X: u16 = 1 << 9;
}

#[repr(C)]
struct GameInfo {
    path: *const c_char,
    data: *const c_void,
    size: usize,
    meta: *const c_char,
}

#[repr(C)]
#[derive(Default)]
struct AvInfo {
    base_width: c_uint,
    base_height: c_uint,
    max_width: c_uint,
    max_height: c_uint,
    aspect_ratio: f32,
    fps: f64,
    sample_rate: f64,
}

#[derive(Default)]
struct Shared {
    /// Latest frame, tightly packed RGB.
    frame: Vec<u8>,
    width: u32,
    height: u32,
    buttons: u16,
    pixel_format: c_uint,
}

thread_local! {
    static SHARED: RefCell<Shared> = RefCell::new(Shared::default());
}

unsafe extern "C" fn cb_environment(cmd: c_uint, data: *mut c_void) -> bool {
    match cmd {
        ENV_SET_PIXEL_FORMAT => unsafe {
            let fmt = *(data as *const c_uint);
            let supported = matches!(fmt, PIXEL_FORMAT_0RGB1555 | PIXEL_FORMAT_XRGB8888 | PIXEL_FORMAT_RGB565);
            if supported {
                SHARED.with(|s| s.borrow_mut().pixel_format = fmt);
            }
            supported
        },
        ENV_GET_SYSTEM_DIRECTORY => unsafe {
            *(data as *mut *const c_char) = c".".as_ptr();
            true
        },
        _ => false,
    }
}

unsafe extern "C" fn cb_video(data: *const c_void, width: c_uint, height: c_uint, pitch: usize) {
    if data.is_null() {
        return; // frame duplication: keep the previous frame
    }
    SHARED.with(|s| {
        let mut s = s.borrow_mut();
        s.width = width;
        s.height = height;
        s.frame.clear();
        let fmt = s.pixel_format;
        let scale5 = |v: u16| ((v << 3) | (v >> 2)) as u8;
        for y in 0..height as usize {
            let row_ptr = unsafe { (data as *const u8).add(y * pitch) };
            if fmt == PIXEL_FORMAT_XRGB8888 {
                let row = unsafe { std::slice::from_raw_parts(row_ptr, width as usize * 4) };
                // XRGB8888 little-endian is B, G, R, X in memory.
                for px in row.chunks_exact(4) {
                    s.frame.extend_from_slice(&[px[2], px[1], px[0]]);
                }
            } else {
                let row = unsafe { std::slice::from_raw_parts(row_ptr, width as usize * 2) };
                for px in row.chunks_exact(2) {
                    let v = u16::from_le_bytes([px[0], px[1]]);
                    let rgb = if fmt == PIXEL_FORMAT_RGB565 {
                        [scale5(v >> 11), (((v >> 5) & 0x3F) << 2 | ((v >> 5) & 0x3F) >> 4) as u8, scale5(v & 0x1F)]
                    } else {
                        [scale5((v >> 10) & 0x1F), scale5((v >> 5) & 0x1F), scale5(v & 0x1F)]
                    };
                    s.frame.extend_from_slice(&rgb);
                }
            }
        }
    });
}

unsafe extern "C" fn cb_audio_sample(_l: i16, _r: i16) {}

unsafe extern "C" fn cb_audio_batch(_data: *const i16, frames: usize) -> usize {
    frames
}

unsafe extern "C" fn cb_input_poll() {}

unsafe extern "C" fn cb_input_state(port: c_uint, device: c_uint, _index: c_uint, id: c_uint) -> i16 {
    if port != 0 || device & 0xFF != DEVICE_JOYPAD {
        return 0;
    }
    SHARED.with(|s| {
        let mask = s.borrow().buttons;
        if id == DEVICE_MASK {
            mask as i16
        } else {
            ((mask >> id) & 1) as i16
        }
    })
}

pub struct Emulator {
    _lib: Library,
    run: unsafe extern "C" fn(),
    reset: unsafe extern "C" fn(),
    unload_game: unsafe extern "C" fn(),
    deinit: unsafe extern "C" fn(),
    serialize_size: unsafe extern "C" fn() -> usize,
    serialize: unsafe extern "C" fn(*mut c_void, usize) -> bool,
    unserialize: unsafe extern "C" fn(*const c_void, usize) -> bool,
    get_memory_data: unsafe extern "C" fn(c_uint) -> *mut c_void,
    get_memory_size: unsafe extern "C" fn(c_uint) -> usize,
    pub fps: f64,
    // Keeps the ROM bytes alive for cores that do not copy them.
    _rom: Vec<u8>,
}

impl Emulator {
    pub fn load(core: &Path, rom: &Path) -> Result<Self, String> {
        let rom_bytes = std::fs::read(rom).map_err(|e| format!("read rom: {e}"))?;
        let rom_path = CString::new(rom.to_string_lossy().as_bytes()).map_err(|e| e.to_string())?;
        unsafe {
            let lib = Library::new(core).map_err(|e| format!("load core: {e}"))?;
            macro_rules! sym {
                ($name:literal, $ty:ty) => {
                    *lib.get::<$ty>($name).map_err(|e| format!("symbol {}: {e}", String::from_utf8_lossy($name)))?
                };
            }
            let set_environment = sym!(b"retro_set_environment", unsafe extern "C" fn(unsafe extern "C" fn(c_uint, *mut c_void) -> bool));
            let set_video = sym!(b"retro_set_video_refresh", unsafe extern "C" fn(unsafe extern "C" fn(*const c_void, c_uint, c_uint, usize)));
            let set_audio = sym!(b"retro_set_audio_sample", unsafe extern "C" fn(unsafe extern "C" fn(i16, i16)));
            let set_audio_batch = sym!(b"retro_set_audio_sample_batch", unsafe extern "C" fn(unsafe extern "C" fn(*const i16, usize) -> usize));
            let set_poll = sym!(b"retro_set_input_poll", unsafe extern "C" fn(unsafe extern "C" fn()));
            let set_state = sym!(b"retro_set_input_state", unsafe extern "C" fn(unsafe extern "C" fn(c_uint, c_uint, c_uint, c_uint) -> i16));
            let init = sym!(b"retro_init", unsafe extern "C" fn());
            let load_game = sym!(b"retro_load_game", unsafe extern "C" fn(*const GameInfo) -> bool);
            let get_av_info = sym!(b"retro_get_system_av_info", unsafe extern "C" fn(*mut AvInfo));

            set_environment(cb_environment);
            init();
            set_video(cb_video);
            set_audio(cb_audio_sample);
            set_audio_batch(cb_audio_batch);
            set_poll(cb_input_poll);
            set_state(cb_input_state);

            let info = GameInfo { path: rom_path.as_ptr(), data: rom_bytes.as_ptr().cast(), size: rom_bytes.len(), meta: std::ptr::null() };
            if !load_game(&info) {
                return Err("retro_load_game failed".into());
            }
            let mut av = AvInfo::default();
            get_av_info(&mut av);

            Ok(Self {
                run: sym!(b"retro_run", unsafe extern "C" fn()),
                reset: sym!(b"retro_reset", unsafe extern "C" fn()),
                unload_game: sym!(b"retro_unload_game", unsafe extern "C" fn()),
                deinit: sym!(b"retro_deinit", unsafe extern "C" fn()),
                serialize_size: sym!(b"retro_serialize_size", unsafe extern "C" fn() -> usize),
                serialize: sym!(b"retro_serialize", unsafe extern "C" fn(*mut c_void, usize) -> bool),
                unserialize: sym!(b"retro_unserialize", unsafe extern "C" fn(*const c_void, usize) -> bool),
                get_memory_data: sym!(b"retro_get_memory_data", unsafe extern "C" fn(c_uint) -> *mut c_void),
                get_memory_size: sym!(b"retro_get_memory_size", unsafe extern "C" fn(c_uint) -> usize),
                fps: av.fps,
                _rom: rom_bytes,
                _lib: lib,
            })
        }
    }

    pub fn set_buttons(&mut self, mask: u16) {
        SHARED.with(|s| s.borrow_mut().buttons = mask);
    }

    pub fn run_frame(&mut self) {
        unsafe { (self.run)() }
    }

    pub fn reset(&mut self) {
        unsafe { (self.reset)() }
    }

    /// Latest frame as `(rgb, width, height)`.
    pub fn frame(&self) -> (Vec<u8>, u32, u32) {
        SHARED.with(|s| {
            let s = s.borrow();
            (s.frame.clone(), s.width, s.height)
        })
    }

    /// The 128 KiB of SNES work RAM, addressed as `$7E0000 + offset`.
    pub fn ram(&self) -> &[u8] {
        unsafe {
            let ptr = (self.get_memory_data)(MEMORY_SYSTEM_RAM) as *const u8;
            let len = (self.get_memory_size)(MEMORY_SYSTEM_RAM);
            if ptr.is_null() { &[] } else { std::slice::from_raw_parts(ptr, len) }
        }
    }

    pub fn save_state(&self) -> Result<Vec<u8>, String> {
        unsafe {
            let mut buf = vec![0u8; (self.serialize_size)()];
            if (self.serialize)(buf.as_mut_ptr().cast(), buf.len()) { Ok(buf) } else { Err("retro_serialize failed".into()) }
        }
    }

    pub fn load_state(&mut self, state: &[u8]) -> Result<(), String> {
        unsafe {
            if (self.unserialize)(state.as_ptr().cast(), state.len()) { Ok(()) } else { Err("retro_unserialize failed".into()) }
        }
    }
}

impl Drop for Emulator {
    fn drop(&mut self) {
        unsafe {
            (self.unload_game)();
            (self.deinit)();
        }
    }
}
