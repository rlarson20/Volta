// The `no_grad` and device-policy guards are `!Send`: each restores thread-local state on
// drop, so dropping one on another thread would change that thread's state instead.
use volta::autograd::no_grad;
use volta::{DevicePolicy, with_device_policy};

fn assert_send<T: Send>(_: T) {}

fn main() {
    assert_send(no_grad());
    assert_send(with_device_policy(DevicePolicy::Strict));
}
