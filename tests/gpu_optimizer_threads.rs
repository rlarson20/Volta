//! #433: a thread that steps a GPU optimizer and exits must not abort the process. The
//! optimizer's unused-state placeholders used to live in a `thread_local!`, whose
//! destructor dropped GPU buffers after wgpu-core's own thread-local could be gone.
//! Run it repeatedly with `just test-repeat-threads "--test gpu_optimizer_threads" 30`.

#![cfg(feature = "gpu")]

mod common;

use volta::nn::SGD;
use volta::storage::Storage;
use volta::{RawTensor, Tensor};

/// One plain SGD step (which binds both placeholders) on a fresh 2x2 GPU weight; returns
/// the updated values.
fn sgd_step_on_gpu() -> Vec<f32> {
    let w: Tensor =
        RawTensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], false).to_device(common::gpu_device());
    w.set_requires_grad(true);
    w.set_grad(Storage::cpu(vec![1.0, -1.0, 0.5, 2.0]));
    SGD::new(vec![w.clone()], 0.5, 0.0, 0.0).step();
    w.data()
}

#[test]
fn a_gpu_step_on_an_exiting_thread_then_on_this_one() {
    if common::skip_if_no_gpu() {
        return;
    }
    let expected = vec![0.5, 2.5, 2.75, 3.0];
    for _ in 0..4 {
        let there = std::thread::spawn(sgd_step_on_gpu)
            .join()
            .expect("the spawned thread exits cleanly");
        assert_eq!(there, expected);
    }
    assert_eq!(sgd_step_on_gpu(), expected);
}
