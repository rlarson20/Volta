use crate::autograd::{GradFn, edge};
use crate::device::Device;
use crate::device_policy::kernel_failed;
use crate::index::fetch;
use crate::ops::gpu_ops::DeviceMax;
use crate::ops::max_rule::{is_max, max_and_count};
use crate::real::{Real, compute_dtype, dispatch_real};
use crate::shape::Shape;
use crate::storage::Storage;
use crate::{RawTensor, Tensor};

/// Reduction operations: reduce tensor to scalar
///
/// These operations collapse all dimensions and require special gradient handling
/// since the output shape differs from the input.
#[derive(Clone, Copy, Debug)]
pub enum ReduceOp {
    /// Sum of all elements; the gradient broadcasts ones.
    Sum,
    /// Maximum element; the gradient goes only to the maximum.
    Max,
    /// Mean of all elements; the gradient broadcasts `1/n`.
    Mean,
}

/// Gradient function for Sum reduction
///
/// Sum reduction collapses to scalar, so gradient broadcasts back to original shape.
#[derive(Clone)]
pub(crate) struct SumGradFn {
    /// Shape of the reduced input.
    input_shape: Shape,
}

impl SumGradFn {
    /// Broadcasts the scalar gradient to the input's shape.
    pub(crate) fn backward(
        &self,
        out_grad: &RawTensor,
        _parents: &[Tensor],
    ) -> Vec<Option<Tensor>> {
        let size: usize = self.input_shape.iter().product();
        // GPU path for a GPU gradient, read on the device; a `None` from the kernel is a
        // failure.
        if out_grad.device.is_gpu() {
            let storage = crate::RawTensor::gpu_sum_backward(&out_grad.data, size)
                .unwrap_or_else(|| kernel_failed("Sum"));
            return vec![Some(RawTensor::new_with_storage(
                storage,
                self.input_shape.clone(),
                out_grad.device.clone(),
                false,
            ))];
        }

        // CPU path, in the gradient's dtype.
        let grad = dispatch_real!(out_grad.data.dtype(), T => {
            Storage::from_compute(out_grad.data.dtype(), vec![fetch(&out_grad.data.host::<T>(), 0); size])
        });
        vec![Some(RawTensor::new_with_storage(
            grad,
            self.input_shape.clone(),
            Device::CPU,
            false,
        ))]
    }
}

/// Gradient function for Max reduction
///
/// As `torch.max()`: the gradient is split evenly over every element equal to the maximum,
/// or over every NaN when the maximum is NaN (S4).
#[derive(Clone)]
pub(crate) struct MaxReduceGradFn {
    /// Shape of the reduced input.
    input_shape: Shape,
    /// Where the maximum was found.
    winners: MaxWinners,
}

/// How a max reduction's backward finds its winners: on the host for a CPU input, by
/// scanning the saved input again, and on the device (never read back) for a GPU input.
#[derive(Clone)]
enum MaxWinners {
    /// Re-scan the parent's values.
    Host,
    /// The maximum and its tie count, in GPU buffers.
    Device(DeviceMax),
}

impl MaxReduceGradFn {
    /// Splits the scalar gradient evenly over the maximum elements; every other element
    /// gets zero.
    pub(crate) fn backward(&self, out_grad: &RawTensor, parents: &[Tensor]) -> Vec<Option<Tensor>> {
        let size: usize = self.input_shape.iter().product();
        // GPU path for a GPU gradient, read on the device; a `None` from the kernel is a
        // failure. The maximum was found on the same device as the gradient lives on.
        if out_grad.device.is_gpu() {
            let MaxWinners::Device(index) = &self.winners else {
                panic!("MaxReduce backward: a GPU gradient for a max found on the CPU");
            };
            let input = parents
                .first()
                .expect("MaxReduce backward: one parent")
                .borrow()
                .data
                .clone();
            let storage = crate::RawTensor::gpu_max_backward(&out_grad.data, &input, size, index)
                .unwrap_or_else(|| kernel_failed("MaxReduce"));
            return vec![Some(RawTensor::new_with_storage(
                storage,
                self.input_shape.clone(),
                out_grad.device.clone(),
                false,
            ))];
        }

        // CPU path, in the gradient's dtype. `saves_input` makes an in-place edit of the
        // input since the forward pass an error, so this scan finds the forward's winners.
        let MaxWinners::Host = self.winners else {
            panic!("MaxReduce backward: a CPU gradient for a max found on the GPU");
        };
        let input = parents
            .first()
            .expect("MaxReduce backward: one parent")
            .borrow()
            .data
            .clone();
        let grad = dispatch_real!(out_grad.data.dtype(), T => {
            let xs = input.host::<T>();
            let (max, count) = max_and_count(&xs);
            let share = fetch(&out_grad.data.host::<T>(), 0) / T::from_usize(count);
            let grad_data: Vec<T> = xs
                .iter()
                .map(|&x| if is_max(x, max) { share } else { T::ZERO })
                .collect();
            Storage::from_compute(out_grad.data.dtype(), grad_data)
        });
        vec![Some(RawTensor::new_with_storage(
            grad,
            self.input_shape.clone(),
            Device::CPU,
            false,
        ))]
    }
}

/// Gradient function for Mean reduction
///
/// Each element gets gradient / `num_elements`.
#[derive(Clone)]
pub(crate) struct MeanGradFn {
    /// Shape of the reduced input.
    input_shape: Shape,
}

impl MeanGradFn {
    /// Broadcasts the scalar gradient divided by the element count.
    pub(crate) fn backward(
        &self,
        out_grad: &RawTensor,
        _parents: &[Tensor],
    ) -> Vec<Option<Tensor>> {
        let size: usize = self.input_shape.iter().product();
        // GPU path for a GPU gradient, read on the device; a `None` from the kernel is a
        // failure.
        if out_grad.device.is_gpu() {
            let storage = crate::RawTensor::gpu_mean_backward(&out_grad.data, size)
                .unwrap_or_else(|| kernel_failed("Mean"));
            return vec![Some(RawTensor::new_with_storage(
                storage,
                self.input_shape.clone(),
                out_grad.device.clone(),
                false,
            ))];
        }

        // CPU path, in the gradient's dtype.
        let grad = dispatch_real!(out_grad.data.dtype(), T => {
            let g = fetch(&out_grad.data.host::<T>(), 0) / T::from_usize(size);
            Storage::from_compute(out_grad.data.dtype(), vec![g; size])
        });
        vec![Some(RawTensor::new_with_storage(
            grad,
            self.input_shape.clone(),
            Device::CPU,
            false,
        ))]
    }
}

// ===== PAIRWISE SUMMATION =====

/// Values [`pairwise_sum_by`] adds in sequence before it stops splitting.
const PAIRWISE_BLOCK: usize = 128;

/// The sum of the `n` values `get(0)..get(n)`, added as a balanced tree whose leaves are
/// sequential runs of up to [`PAIRWISE_BLOCK`] values (numpy's pairwise summation).
///
/// The rounding error grows as `eps * (PAIRWISE_BLOCK + log2 n)` instead of `eps * n`, and a
/// sum of 2^25 ones is exact, where a sequential f32 chain stalls at 2^24 (S6). The first
/// value starts each run (no `0 + x`), so a sum of one `-0.0` stays `-0.0`. An empty sum is
/// zero. `get` reads one element, so a strided reduction passes its own index arithmetic.
pub(crate) fn pairwise_sum_by<T: Real>(n: usize, get: impl Fn(usize) -> T) -> T {
    /// The sum of `get(lo)..get(hi)`, for `lo < hi`.
    fn range<T: Real>(lo: usize, hi: usize, get: &impl Fn(usize) -> T) -> T {
        if hi - lo <= PAIRWISE_BLOCK {
            ((lo + 1)..hi).fold(get(lo), |acc, i| acc + get(i))
        } else {
            let mid = lo + (hi - lo) / 2;
            range(lo, mid, get) + range(mid, hi, get)
        }
    }
    if n == 0 { T::ZERO } else { range(0, n, &get) }
}

/// The pairwise sum of a slice ([`pairwise_sum_by`]).
pub(crate) fn pairwise_sum<T: Real>(xs: &[T]) -> T {
    pairwise_sum_by(xs.len(), |i| fetch(xs, i))
}

// ===== REDUCE OPERATIONS =====

impl RawTensor {
    /// Apply a reduction operation that collapses tensor to scalar
    ///
    /// All reduction ops produce a shape \[1\] output.
    /// # Panics
    /// unwrap map
    #[must_use]
    pub fn reduce_op(self_t: &Tensor, op: ReduceOp) -> Tensor {
        let dtype = compute_dtype(|| format!("{op:?}"), &[self_t.dtype()]);
        let (data, shape, device) = {
            let s = self_t.borrow();
            (s.data.clone(), s.shape.clone(), s.device.clone())
        };

        // On the GPU, a `None` from the kernel is a failure. The reduction runs there and its
        // scalar stays there (rounded to the input's dtype on the device, as the CPU rounds a
        // half reduction once): nothing is read back.
        let (storage, grad_fn): (Storage, GradFn) = match op {
            ReduceOp::Sum => {
                let sum = if device.is_gpu() {
                    Self::gpu_sum_reduce(&data).unwrap_or_else(|| kernel_failed("Sum"))
                } else {
                    dispatch_real!(dtype, T => {
                        Storage::from_compute(dtype, vec![pairwise_sum(&data.host::<T>())])
                    })
                };
                (sum, GradFn::Sum(SumGradFn { input_shape: shape }))
            }
            ReduceOp::Max => {
                let (max, winners) = if device.is_gpu() {
                    let (max, index) =
                        Self::gpu_max_reduce(&data).unwrap_or_else(|| kernel_failed("MaxReduce"));
                    (max, MaxWinners::Device(index))
                } else {
                    dispatch_real!(dtype, T => {
                        let (max_val, _) = max_and_count(&data.host::<T>());
                        (Storage::from_compute(dtype, vec![max_val]), MaxWinners::Host)
                    })
                };
                (
                    max,
                    GradFn::MaxReduce(MaxReduceGradFn {
                        input_shape: shape,
                        winners,
                    }),
                )
            }
            ReduceOp::Mean => {
                let mean = if device.is_gpu() {
                    Self::gpu_mean_reduce(&data).unwrap_or_else(|| kernel_failed("Mean"))
                } else {
                    dispatch_real!(dtype, T => {
                        let sum: T = pairwise_sum(&data.host::<T>());
                        Storage::from_compute(dtype, vec![sum / T::from_usize(data.len())])
                    })
                };
                (mean, GradFn::Mean(MeanGradFn { input_shape: shape }))
            }
        };

        edge(
            [self_t.clone()],
            storage,
            Shape::derived(vec![1]),
            device,
            grad_fn,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::fetch;

    /// Lengths either side of every point where the tree splits.
    const LENGTHS: [usize; 9] = [0, 1, 2, 127, 128, 129, 255, 256, 257];

    #[test]
    fn pairwise_sum_of_ones_is_the_count_at_every_split_point() {
        for n in LENGTHS {
            assert_eq!(pairwise_sum(&vec![1.0f32; n]), n as f32, "n={n}");
            assert_eq!(pairwise_sum(&vec![1.0f64; n]), n as f64, "n={n}");
        }
    }

    #[test]
    fn pairwise_sum_of_2_pow_25_ones_is_exact_in_f32() {
        // A sequential f32 chain stalls at 2^24 (S6).
        let ones = vec![1.0f32; 1 << 25];
        assert_eq!(pairwise_sum(&ones), 33_554_432.0);
    }

    #[test]
    fn pairwise_sum_keeps_terms_a_running_sum_loses() {
        // 1.0, then 100000 terms each below half an ulp of 1.0: a running sum stays 1.0.
        let mut xs = vec![3.0e-8f32; 100_001];
        *crate::index::slot(&mut xs, 0) = 1.0;
        let sequential = xs.iter().fold(0.0f32, |a, &b| a + b);
        assert_eq!(sequential, 1.0);
        let want = 1.0 + 100_000.0 * 3.0e-8_f64;
        let got = f64::from(pairwise_sum(&xs));
        // The first leaf run (128 values) still adds onto the 1.0 in sequence and drops its
        // 127 terms, about 4e-6: the error is bounded by the block, not by n (3e-3 above).
        assert!((got - want).abs() < 1e-5, "pairwise {got}, exact {want}");
    }

    #[test]
    fn pairwise_sum_by_reads_each_index_once_in_range() {
        let seen = std::cell::RefCell::new(vec![0u32; 1000]);
        let total: f32 = pairwise_sum_by(1000, |i| {
            *crate::index::slot(&mut seen.borrow_mut(), i) += 1;
            1.0
        });
        assert_eq!(total, 1000.0);
        assert!(seen.borrow().iter().all(|&c| c == 1));
    }

    #[test]
    fn pairwise_sum_of_empty_is_zero_and_of_one_negative_zero_is_negative_zero() {
        assert_eq!(pairwise_sum::<f32>(&[]), 0.0);
        let z = pairwise_sum(&[-0.0f32]);
        assert!(z == 0.0 && z.is_sign_negative());
        assert_eq!(fetch(&[2.5f32], 0), pairwise_sum(&[2.5f32]));
    }
}
