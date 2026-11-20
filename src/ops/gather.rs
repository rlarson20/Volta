//! `gather` along one axis, as `torch.gather` (#78).
//!
//! `out[i_0, .., i_axis, .., i_n] = input[i_0, .., index[i_0, .., i_n], .., i_n]`: `index`
//! has the input's rank, the output has `index`'s shape, and on every other axis `index`
//! may be shorter than the input (it reads a prefix there), as in PyTorch. The backward
//! scatter-adds the gradient into the positions that were read, summing repeats.

use crate::autograd::{GradFn, edge};
use crate::device::Device;
use crate::dtype::DType;
use crate::error::{Result, VoltaError};
use crate::index::{fetch, slot};
use crate::real::{Real, dispatch_bits, dispatch_real};
use crate::shape::{Axis, Shape, Strides};
use crate::storage::Storage;
use crate::tensor::{RawTensor, Tensor};

/// Backward of [`Tensor::gather`]: scatter-add into the input's shape.
#[derive(Clone)]
pub(crate) struct GatherGradFn {
    /// The input's shape: the gradient's shape.
    input_shape: Shape,
    /// For each output element, in row-major order, the flat input offset it was read from.
    sources: Vec<usize>,
}

impl GatherGradFn {
    /// Adds each output gradient into the input element it was read from; an element read
    /// twice gets both contributions, one never read gets zero.
    pub(crate) fn backward(
        &self,
        out_grad: &RawTensor,
        _parents: &[Tensor],
    ) -> Vec<Option<Storage>> {
        let dtype = out_grad.data.dtype();
        let grad = dispatch_real!(dtype, T => {
            let g = out_grad.data.host::<T>();
            let mut acc = vec![T::ZERO; self.input_shape.numel()];
            for (i, &src) in self.sources.iter().enumerate() {
                *slot(&mut acc, src) += fetch(&g, i);
            }
            Storage::from_compute(dtype, acc)
        });
        vec![Some(grad)]
    }
}

/// The flat input offset of every output element of `gather(input_dims, axis, index)`,
/// with every argument checked. `index` holds the raw values.
///
/// # Errors
/// See [`Tensor::try_gather`].
fn gather_sources(
    input_dims: &[usize],
    axis: Axis,
    index_dims: &[usize],
    index: &[i64],
) -> Result<Vec<usize>> {
    let rank = input_dims.len();
    let Axis(ax) = axis;
    if ax >= rank {
        return Err(VoltaError::DimensionOutOfBounds {
            dim: ax,
            shape: input_dims.to_vec(),
        });
    }
    if index_dims.len() != rank {
        return Err(VoltaError::InvalidParameter(format!(
            "gather: index has rank {} ({index_dims:?}), the input rank {rank} ({input_dims:?})",
            index_dims.len()
        )));
    }
    for (d, (&n, &m)) in index_dims.iter().zip(input_dims).enumerate() {
        if d != ax && n > m {
            return Err(VoltaError::InvalidParameter(format!(
                "gather: index {index_dims:?} is longer than the input {input_dims:?} on axis \
                 {d} (only axis {ax} may differ freely)"
            )));
        }
    }
    let extent = fetch(input_dims, ax);
    let in_strides = Strides::from_shape(input_dims);
    let in_strides = in_strides.truncate_rank();
    let idx_strides = Strides::from_shape(index_dims);
    let idx_strides = idx_strides.truncate_rank();

    let mut sources = Vec::with_capacity(index.len());
    for (i, &v) in index.iter().enumerate() {
        let Some(k) = usize::try_from(v).ok().filter(|&k| k < extent) else {
            return Err(VoltaError::InvalidParameter(format!(
                "gather: index {v} out of range for axis {ax} of size {extent}"
            )));
        };
        // The coordinates of output element `i` (row-major over `index_dims`), with axis
        // `ax` replaced by the index value.
        let mut rem = i;
        let mut src = 0;
        for d in 0..rank {
            let stride = fetch(idx_strides, d);
            let coord = rem / stride;
            rem %= stride;
            let c = if d == ax { k } else { coord };
            src += c * fetch(in_strides, d);
        }
        sources.push(src);
    }
    Ok(sources)
}

impl RawTensor {
    /// [`Tensor::try_gather`].
    ///
    /// # Errors
    /// As [`Tensor::try_gather`].
    pub(crate) fn try_gather_op(self_t: &Tensor, axis: Axis, index: &Tensor) -> Result<Tensor> {
        let index_values: Vec<i64> = match index.dtype() {
            DType::I64 => index.to_vec_as::<i64>()?,
            DType::I32 => index
                .to_vec_as::<i32>()?
                .into_iter()
                .map(i64::from)
                .collect(),
            other @ (DType::U8
            | DType::Bool
            | DType::F16
            | DType::BF16
            | DType::F32
            | DType::F64) => {
                return Err(VoltaError::DTypeMismatch {
                    expected: DType::I64,
                    actual: other,
                });
            }
        };
        let input_shape = self_t.shape();
        let index_shape = index.shape();
        let sources = gather_sources(
            input_shape.truncate_rank(),
            axis,
            index_shape.truncate_rank(),
            &index_values,
        )?;

        // A copy of each element's bits: any dtype, no arithmetic. GPU data is read back
        // (no kernel), and `edge()` applies the device policy to the CPU result.
        let data = self_t.borrow().data.to_device(&Device::CPU);
        let dtype = data.dtype();
        let out = dispatch_bits!(dtype, B => {
            let bits = data.bits::<B>();
            let picked: Vec<B> = sources.iter().map(|&s| fetch(bits, s)).collect();
            Storage::from_bits(dtype, &picked)
        });
        Ok(edge(
            [self_t.clone()],
            out,
            Shape::derived(index_shape.truncate_rank().to_vec()),
            Device::CPU,
            GradFn::Gather(GatherGradFn {
                input_shape,
                sources,
            }),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An index tensor of `shape`.
    fn idx(v: &[i64], shape: &[usize]) -> Tensor {
        RawTensor::from_typed(v.to_vec(), shape)
    }

    /// torch: `torch.gather(torch.tensor([[1., 2.], [3., 4.]]), 1, torch.tensor([[0, 0],
    /// [1, 0]]))` is `[[1., 1.], [4., 3.]]` (the example in torch's docs).
    #[test]
    fn gather_axis_1_matches_torch_docs_example() {
        let t = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], false);
        let out = t.gather(Axis(1), &idx(&[0, 0, 1, 0], &[2, 2]));
        assert_eq!(out.shape().truncate_rank(), &[2, 2]);
        assert_eq!(out.data(), vec![1.0, 1.0, 4.0, 3.0]);
    }

    /// Axis 0 on a `[3, 2]` input with a shorter `[2, 2]` index, and axis 1 with a
    /// one-column index (the class-index pick of `nll_loss`).
    #[test]
    fn gather_axis_0_and_a_single_column() {
        let t = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], false);
        let out = t.gather(Axis(0), &idx(&[2, 0, 1, 2], &[2, 2]));
        assert_eq!(out.data(), vec![5.0, 2.0, 3.0, 6.0]);
        let col = t.gather(Axis(1), &idx(&[1, 0, 1], &[3, 1]));
        assert_eq!(col.shape().truncate_rank(), &[3, 1]);
        assert_eq!(col.data(), vec![2.0, 3.0, 6.0]);
    }

    /// A rank-3 gather on the middle axis, with I32 indices, against the formula.
    #[test]
    fn gather_rank_3_middle_axis_i32() {
        let data: Vec<f32> = (0..24).map(|v| v as f32).collect();
        let t = RawTensor::new(data, &[2, 3, 4], false);
        let index: Vec<i32> = vec![2, 0, 1, 1, 0, 2, 2, 0];
        let out = t.gather(Axis(1), &RawTensor::from_typed(index.clone(), &[2, 1, 4]));
        let want: Vec<f32> = (0..2)
            .flat_map(|i| (0..4).map(move |k| (i, k)))
            .map(|(i, k)| {
                let j = usize::try_from(fetch(&index, i * 4 + k)).expect("non-negative");
                (i * 12 + j * 4 + k) as f32
            })
            .collect();
        assert_eq!(out.data(), want);
    }

    /// The gradient is a scatter-add: repeated picks sum, unread elements get zero.
    #[test]
    fn gather_gradient_scatter_adds_repeats() {
        let t = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], true);
        let out = t.gather(Axis(1), &idx(&[2, 2, 0, 1], &[2, 2]));
        let w = RawTensor::new(vec![1.0, 10.0, 100.0, 1000.0], &[2, 2], false);
        out.elem_mul(&w)
            .sum()
            .into_loss()
            .expect("scalar")
            .backward();
        assert_eq!(
            t.grad().expect("a gradient"),
            vec![0.0, 0.0, 11.0, 100.0, 1000.0, 0.0]
        );
    }

    /// Seeded random input, random index (repeats included) and a random weighted loss,
    /// on a rank-3 input along each axis, with a shorter index on the other axes.
    #[test]
    fn gather_gradient_check_random_every_axis() {
        crate::manual_seed(0x78);
        for axis in 0..3 {
            // f64: gather is linear, so the finite difference only measures rounding, which
            // f32 at this loss scale puts above the tolerance on small gradients.
            let t = RawTensor::randn(&[3, 4, 2]).to_dtype(DType::F64);
            t.set_requires_grad(true);
            let mut index_dims = vec![2, 3, 2];
            *slot(&mut index_dims, axis) = 5;
            let extent = fetch(&[3, 4, 2], axis);
            let n: usize = index_dims.iter().product();
            let values: Vec<i64> = (0..n)
                .map(|i| i64::try_from((i * 7 + 3) % extent).expect("small"))
                .collect();
            let index = idx(&values, &index_dims);
            let w = RawTensor::randn(&index_dims).to_dtype(DType::F64);
            let (max_err, mean_err, passed) = RawTensor::check_gradients(
                &t,
                |x| x.gather(Axis(axis), &index).elem_mul(&w).sum(),
                1e-6,
                1e-6,
            );
            assert!(
                passed,
                "axis {axis}: max_error={max_err:.3e}, mean_error={mean_err:.3e}"
            );
        }
    }

    /// Any dtype gathers (bits are copied); the result keeps it.
    #[test]
    fn gather_copies_integer_and_f64_values() {
        let ints = RawTensor::from_typed(vec![7_i64, 8, 9], &[3]);
        let out = ints.gather(Axis(0), &idx(&[2, 2, 0], &[3]));
        assert_eq!(out.to_vec_as::<i64>().expect("I64"), vec![9, 9, 7]);
        let wide = RawTensor::new(vec![0.5, 1.5], &[2], false).to_dtype(DType::F64);
        let out = wide.gather(Axis(0), &idx(&[1], &[1]));
        assert_eq!(out.dtype(), DType::F64);
        assert_eq!(out.to_vec_as::<f64>().expect("F64"), vec![1.5]);
    }

    /// Each bad argument is an error naming it, before anything is computed.
    #[test]
    fn try_gather_rejects_bad_arguments() {
        let t = RawTensor::new(vec![1.0; 6], &[2, 3], false);
        let err = |axis, index: &Tensor| match t.try_gather(Axis(axis), index) {
            Ok(_) => panic!("axis {axis}: expected an error"),
            Err(e) => e.to_string(),
        };
        assert!(err(2, &idx(&[0, 0], &[2, 1])).contains("Dimension 2 out of bounds"));
        assert!(err(1, &idx(&[0, 0], &[2])).contains("rank 1"));
        assert!(err(1, &idx(&[0, 0, 0], &[3, 1])).contains("longer than the input"));
        assert!(err(1, &idx(&[0, 3], &[2, 1])).contains("index 3 out of range"));
        assert!(err(1, &idx(&[0, -1], &[2, 1])).contains("index -1 out of range"));
        let float_index = RawTensor::new(vec![0.0, 1.0], &[2, 1], false);
        assert!(err(1, &float_index).contains("DType mismatch"));
    }
}
