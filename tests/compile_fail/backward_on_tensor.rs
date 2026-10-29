// Backward only exists on `ScalarLoss`, so a bare `Tensor` can't start it.
use volta::RawTensor;

fn main() {
    let x = RawTensor::new(vec![1.0, 2.0], &[2], true);
    x.backward();
}
