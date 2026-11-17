//! `randn_like` of a GPU tensor is on that GPU (#99). Prints `GPU-RAN` before its
//! assertions, because without an adapter the test returns early and still reports `ok`.

#![cfg(feature = "gpu")]

#[path = "common/mod.rs"]
mod common;

use common::{gpu_device, skip_if_no_gpu};
use volta::{RawTensor, randn_like};

#[test]
fn randn_like_keeps_the_gpu_device() {
    if skip_if_no_gpu() {
        return;
    }
    println!("GPU-RAN randn_like_keeps_the_gpu_device");
    let g = RawTensor::zeros(&[3]).to_device(gpu_device());
    let r = randn_like(&g);
    assert_eq!(r.device(), g.device());
}
