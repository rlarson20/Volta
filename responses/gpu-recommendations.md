**Context**

As part of the Phase 2 push to make Volta’s core ops device-aware, the GPU dispatch must now be guarded by a unified check that all inputs share the same GPU device before invoking `gpu_ops`.
While some ops already touched GPU kernels, the dispatch logic wasn’t consistent or future-proof, and mixed-device scenarios could accidentally trigger GPU paths that immediately fall back to CPU without explicit awareness.
We want to centralize the “same GPU device” check, wire it through binary/unary/matmul, and add a regression test that proofs the automatic CPU fallback when tensors live on different devices.

---

**Plan**

1. Add a reusable helper on `RawTensor` that returns the common `Device` when every operand is GPU-backed and on the same GPU; this helper will be used by all GPU-capable ops.
2. Update the GPU dispatch guards inside `ops/binary.rs`, `ops/unary.rs`, and the 2D `matmul` branch so they only attempt GPU kernels when that helper succeeds. This makes the dispatch logic explicit and future-friendly.
3. Add a GPU smoke test asserting that mixed-device binary ops gracefully stay on CPU (auto fallback) while still producing correct results. This documents the new dispatch guarantee.

---

**Patches**

> **src/tensor.rs**

```diff
@@
 impl RawTensor {
     /// Create a new tensor from data and shape
     ///
@@
         Self::new(data, shape, false)
     }
 }
+
+#[cfg(feature = "gpu")]
+impl RawTensor {
+    /// Return the GPU device shared by `tensors` if every tensor lives on the same GPU.
+    ///
+    /// This avoids accidentally invoking GPU kernels when inputs are on mixed devices.
+    pub(crate) fn common_gpu_device(tensors: &[&Tensor]) -> Option<Device> {
+        let first = tensors.first()?;
+        let first_device = first.borrow().device.clone();
+        if !first_device.is_gpu() {
+            return None;
+        }
+        for tensor in tensors.iter().skip(1) {
+            let device = tensor.borrow().device.clone();
+            if device != first_device {
+                return None;
+            }
+        }
+        Some(first_device)
+    }
+}
```

> **src/ops/binary.rs**

```diff
         // If both operands are already on the same GPU and we have a matching
         // kernel, try to perform the operation there and fall back to CPU
         // otherwise.
         #[cfg(feature = "gpu")]
         {
-            if let Some((shape, storage, device)) = Self::try_gpu_binary_result(self_t, other, op) {
+            if RawTensor::common_gpu_device(&[self_t, other]).is_some() {
+                if let Some((shape, storage, device)) =
+                    Self::try_gpu_binary_result(self_t, other, op)
+                {
                     let out = Rc::new(RefCell::new(RawTensor {
                         data: storage,
                         shape,
                         grad: None,
                         requires_grad,
                         grad_fn: None,
                         parents: vec![self_t.clone(), other.clone()],
                         device,
                     }));
                     if requires_grad {
```

> **src/ops/unary.rs**

```diff
         // Fast path: if the tensor already lives on GPU and we have a matching
         // kernel, try to execute the op there.
-        #[cfg(feature = "gpu")]
-        {
-            if device.is_gpu()
-                && let Some(kernel) = unary_kernel_name(op)
-                && let Some(storage) = RawTensor::gpu_unary(&data, kernel)
-            {
+        #[cfg(feature = "gpu")]
+        {
+            if RawTensor::common_gpu_device(&[t]).is_some()
+                && let Some(kernel) = unary_kernel_name(op)
+                && let Some(storage) = RawTensor::gpu_unary(&data, kernel)
+            {
                 let out = Rc::new(RefCell::new(RawTensor {
                     data: storage,
                     shape: shape.clone(),
                     grad: None,
                     requires_grad: req,
                     grad_fn: None,
                     parents: vec![t.clone()],
                     device,
                 }));
```

> **src/ops/matmul.rs**

```diff
                 #[cfg(feature = "gpu")]
                 {
-                    if dev_a.is_gpu() && dev_b.is_gpu() && dev_a == dev_b {
+                    if let Some(device) = RawTensor::common_gpu_device(&[self_t, other]) {
                         if let Some(storage) = Self::gpu_matmul(&data_a, &data_b, m, n, p) {
                             let requires_grad = req_a || req_b;
                             let out = Rc::new(RefCell::new(RawTensor {
                                 data: storage,
                                 shape: vec![m, p],
                                 grad: None,
                                 requires_grad,
                                 grad_fn: None,
                                 parents: vec![self_t.clone(), other.clone()],
-                                device: dev_a.clone(),
+                                device: device.clone(),
                             }));
                             if requires_grad {
                                 out.borrow_mut().grad_fn = Some(Box::new(MatMulGradFn));
                             }
                             return out;
```

> **tests/gpu_smoke_test.rs**

```diff
 #[test]
 #[cfg(feature = "gpu")]
 fn test_tensor_gpu_unary_relu_matches_cpu() {
@@
     }
 }

+#[test]
+#[cfg(feature = "gpu")]
+fn test_tensor_gpu_mixed_device_binary_fallbacks_to_cpu() {
+    use volta::{Device, is_gpu_available};
+    use volta::TensorOps;
+
+    if !is_gpu_available() {
+        return;
+    }
+
+    let dev = Device::GPU("TestDevice".to_string());
+    let a = RawTensor::new(vec![1.0, 2.0], &[2], false).to_device(dev.clone());
+    let b = RawTensor::new(vec![3.0, 4.0], &[2], false);
+
+    let z = a.add(&b);
+    {
+        let zb = z.borrow();
+        assert!(
+            zb.device.is_cpu(),
+            "Mixed device binary ops should fall back to CPU storage"
+        );
+        assert_eq!(zb.data.to_vec(), vec![4.0, 6.0]);
+    }
+}
+
 #[test]
 #[cfg(feature = "gpu")]
 fn test_tensor_gpu_binary_basic_matches_cpu() {
```

---

**Tests**

- Added `test_tensor_gpu_mixed_device_binary_fallbacks_to_cpu` to confirm that mixed-device inputs safely fall back to CPU computation while remaining numerically correct.
- Recommended verification: run `cargo test` and `cargo test --features gpu` to ensure no regressions in either profile.

---

**Verification checklist**

- [ ] CPU-only build compiles (`cargo test`)
- [ ] GPU build compiles (`cargo test --features gpu`)
- [ ] GPU tests pass when a GPU is available
- [ ] Automatic GPU/CPU dispatch produces numerically identical results
- [ ] Mixed-device fallbacks exercise the CPU path (new regression test covers this)

---

**Performance note**

Dispatch changes only gate existing kernels; there’s no new computation, so performance is unchanged except for avoiding unnecessary GPU calls when inputs are on different devices.

---

**Roadmap update**

- **Phase 2 progress:** Added a reusable “same GPU device” guard and wired it through binary/unary/matmul dispatch logic plus a regression test for mixed-device fallbacks. This makes GPU dispatch explicit and reliable.
- **Next target:** Phase 3—extend gradient functions to be device-aware so backward passes can stay on GPU (start with unary/binary grads).
