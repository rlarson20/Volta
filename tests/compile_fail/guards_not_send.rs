// The `no_grad`, device-policy and dtype-policy guards are `!Send`: each restores
// thread-local state on drop, so dropping one on another thread would change that thread's
// state instead.
use volta::autograd::no_grad;
use volta::{DTypePolicy, DevicePolicy, with_device_policy, with_dtype_policy};

fn assert_send<T: Send>(_: T) {}

fn main() {
    assert_send(no_grad());
    assert_send(with_device_policy(DevicePolicy::Strict));
    assert_send(with_dtype_policy(DTypePolicy::Strict));
}
