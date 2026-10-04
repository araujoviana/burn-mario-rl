//! Backend chosen at build time by cargo feature (cuda > wgpu > tch > flex > ndarray).

use burn::backend::Autodiff;

#[cfg(feature = "cuda")]
pub type Inner = burn::backend::Cuda;
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
pub type Inner = burn::backend::Wgpu;
#[cfg(all(feature = "tch", not(any(feature = "cuda", feature = "wgpu"))))]
pub type Inner = burn::backend::LibTorch;
#[cfg(all(feature = "flex", not(any(feature = "cuda", feature = "wgpu", feature = "tch"))))]
pub type Inner = burn::backend::Flex;
#[cfg(not(any(feature = "cuda", feature = "wgpu", feature = "tch", feature = "flex")))]
pub type Inner = burn::backend::NdArray;

pub type Train = Autodiff<Inner>;

pub fn name() -> &'static str {
    std::any::type_name::<Inner>()
}
