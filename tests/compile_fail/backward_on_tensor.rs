// Backward only exists on `ScalarLoss`, so a bare `Tensor` can't start it.
// `TensorOps` stays imported so this fails if the trait method ever comes back.
use volta::RawTensor;
#[allow(unused_imports)]
use volta::TensorOps;

fn main() {
    let x = RawTensor::new(vec![1.0, 2.0], &[2], true);
    x.backward();
}
