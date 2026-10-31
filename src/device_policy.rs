//! What happens when an op can't run on its inputs' device (ADDENDUM §2.1).
//!
//! A tensor's device label is a promise: an op's output lives on its inputs' device,
//! and nothing moves silently. Three things break that promise, and each is handled
//! here:
//!
//! - **Mixed devices.** Inputs on different devices panic under every policy.
//! - **A gap**: a GPU input reaches an op, or a case of an op, that has no GPU kernel.
//!   [`DevicePolicy::Strict`] panics. [`DevicePolicy::PreferGpu`] runs the op on the
//!   CPU, moves the result back to the GPU, and records a [`FallbackEvent`].
//! - **A failure**: a GPU kernel exists and returns no result. That panics under every
//!   policy, because retrying on the CPU would hide a broken kernel.
//!
//! The policy is thread-local and set for a scope with [`with_device_policy`], the
//! same shape as [`crate::autograd::no_grad`].

use std::cell::{Cell, RefCell};
use std::collections::HashSet;
use std::marker::PhantomData;

use crate::Device;

/// How ops treat a GPU input that has no GPU kernel.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DevicePolicy {
    /// A gap panics. This is the default.
    #[default]
    Strict,
    /// A gap runs on the CPU, its result moves back to the GPU, and a
    /// [`FallbackEvent`] is recorded. Mixed devices and kernel failures still panic.
    PreferGpu,
}

/// Which pass a fallback happened in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pass {
    /// An op's forward computation.
    Forward,
    /// A gradient computed in `backward()`.
    Backward,
}

/// One op that ran on the CPU for GPU inputs under [`DevicePolicy::PreferGpu`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FallbackEvent {
    /// The op, as `GradFn` names it (`"Binary(Add)"`, `"MatMul"`, ...).
    pub op: String,
    /// The pass the fallback happened in.
    pub pass: Pass,
}

thread_local! {
    static POLICY: Cell<DevicePolicy> = const { Cell::new(DevicePolicy::Strict) };
    static EVENTS: RefCell<Vec<FallbackEvent>> = const { RefCell::new(Vec::new()) };
    // Ops that have already logged a fallback warning on this thread.
    static WARNED: RefCell<HashSet<String>> = RefCell::new(HashSet::new());
}

/// This thread's current device policy.
#[must_use]
pub fn device_policy() -> DevicePolicy {
    POLICY.with(Cell::get)
}

/// Set this thread's device policy until the returned guard drops.
///
/// The guard restores the policy that was active before it, so scopes nest, and it
/// restores it on unwind too.
#[must_use]
pub fn with_device_policy(policy: DevicePolicy) -> DevicePolicyGuard {
    let prev = POLICY.with(|p| p.replace(policy));
    DevicePolicyGuard {
        prev,
        _not_send: PhantomData,
    }
}

/// Restores the previous device policy on drop. Made by [`with_device_policy`].
///
/// `!Send`: the policy is thread-local, so a guard dropped on another thread would set
/// that thread's policy and leave this one's changed. `tests/compile_fail/guards_not_send.rs`
/// pins it.
pub struct DevicePolicyGuard {
    /// The policy to restore.
    prev: DevicePolicy,
    /// Makes the guard `!Send` (and `!Sync`).
    _not_send: PhantomData<*const ()>,
}

impl Drop for DevicePolicyGuard {
    fn drop(&mut self) {
        POLICY.with(|p| p.set(self.prev));
    }
}

/// Drain and return the fallbacks recorded on this thread, oldest first.
#[must_use]
pub fn take_fallback_events() -> Vec<FallbackEvent> {
    EVENTS.with(|e| std::mem::take(&mut *e.borrow_mut()))
}

/// Apply the policy to a gap: `op` has no GPU kernel for GPU inputs.
///
/// # Panics
/// Under [`DevicePolicy::Strict`].
pub(crate) fn on_gap(op: &str, pass: Pass) {
    match device_policy() {
        DevicePolicy::Strict => panic!(
            "{op}: no GPU kernel for this case; wrap the call in \
             with_device_policy(DevicePolicy::PreferGpu) to run it on the CPU"
        ),
        DevicePolicy::PreferGpu => {
            if WARNED.with(|w| w.borrow_mut().insert(op.to_string())) {
                log::warn!("{op}: no GPU kernel; running on the CPU (DevicePolicy::PreferGpu)");
            }
            EVENTS.with(|e| {
                e.borrow_mut().push(FallbackEvent {
                    op: op.to_string(),
                    pass,
                });
            });
        }
    }
}

/// A GPU kernel exists for this case and returned no result.
///
/// # Panics
/// Always, under every policy.
pub(crate) fn kernel_failed(op: &str) -> ! {
    panic!("{op}: GPU kernel returned no result")
}

/// The one device all `devices` share, or `None` if there are none.
///
/// # Panics
/// If two devices differ. GPU names count, so this stays correct once there is more
/// than one GPU.
pub(crate) fn common_device<'a>(
    op: &str,
    devices: impl IntoIterator<Item = &'a Device>,
) -> Option<Device> {
    let mut devices = devices.into_iter();
    let first = devices.next()?;
    if let Some(other) = devices.find(|d| *d != first) {
        panic!("{op}: expected all inputs on one device, found {first} and {other}");
    }
    Some(first.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `Strict` is the documented default.
    #[test]
    fn default_is_strict() {
        assert_eq!(DevicePolicy::default(), DevicePolicy::Strict);
    }

    /// A thread with no guard runs under `Strict`.
    #[test]
    fn thread_starts_strict() {
        let policy = std::thread::spawn(device_policy).join().unwrap();
        assert_eq!(policy, DevicePolicy::Strict);
    }

    /// Dropping an inner guard restores the outer scope's policy.
    #[test]
    fn guards_nest_and_restore() {
        let _outer = with_device_policy(DevicePolicy::Strict);
        {
            let _inner = with_device_policy(DevicePolicy::PreferGpu);
            assert_eq!(device_policy(), DevicePolicy::PreferGpu);
        }
        assert_eq!(device_policy(), DevicePolicy::Strict);
    }

    /// A panic inside a scope still restores the previous policy.
    #[test]
    fn guard_restores_on_panic() {
        let _outer = with_device_policy(DevicePolicy::Strict);
        let result = std::panic::catch_unwind(|| {
            let _g = with_device_policy(DevicePolicy::PreferGpu);
            panic!("inside a PreferGpu scope");
        });
        assert!(result.is_err());
        assert_eq!(device_policy(), DevicePolicy::Strict);
    }

    /// Under `Strict`, a gap panics with a message naming the op and the opt-in.
    #[test]
    #[should_panic(expected = "Foo: no GPU kernel for this case")]
    fn strict_gap_panics() {
        let _g = with_device_policy(DevicePolicy::Strict);
        on_gap("Foo", Pass::Forward);
    }

    /// Under `PreferGpu`, gaps are recorded in order and draining empties the log.
    #[test]
    fn prefer_gpu_gap_records_and_drains() {
        let _g = with_device_policy(DevicePolicy::PreferGpu);
        let _ = take_fallback_events();
        on_gap("Foo", Pass::Forward);
        on_gap("Bar", Pass::Backward);
        assert_eq!(
            take_fallback_events(),
            vec![
                FallbackEvent {
                    op: "Foo".to_string(),
                    pass: Pass::Forward
                },
                FallbackEvent {
                    op: "Bar".to_string(),
                    pass: Pass::Backward
                },
            ]
        );
        assert!(take_fallback_events().is_empty());
    }

    /// CPU and GPU inputs together panic.
    #[test]
    #[should_panic(expected = "Foo: expected all inputs on one device")]
    fn mixed_devices_panic() {
        let _ = common_device("Foo", [&Device::CPU, &Device::GPU("x".to_string())]);
    }

    /// Two different GPUs are different devices.
    #[test]
    #[should_panic(expected = "Foo: expected all inputs on one device")]
    fn different_gpus_panic() {
        let _ = common_device(
            "Foo",
            [&Device::GPU("a".to_string()), &Device::GPU("b".to_string())],
        );
    }
}
