// The `no_grad`, device-policy, dtype-policy and autocast guards are `!Send`: each restores
// thread-local state on drop, so dropping one on another thread would change that thread's
// state instead.
use volta::amp::autocast;
use volta::autograd::no_grad;
use volta::{DType, DTypePolicy, DevicePolicy, with_device_policy, with_dtype_policy};

fn assert_send<T: Send>(_: T) {}

fn main() {
    assert_send(no_grad());
    assert_send(with_device_policy(DevicePolicy::Strict));
    assert_send(with_dtype_policy(DTypePolicy::Strict));
    assert_send(autocast(DType::BF16));
}
