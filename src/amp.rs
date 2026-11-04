//! Automatic mixed precision (ADDENDUM §11 P7): [`autocast`] and [`GradScaler`].
//!
//! Inside an [`autocast`] scope, the ops PyTorch's CPU autocast runs in low precision
//! (`matmul`, `Linear`, `Conv2d`, `ConvTranspose2d`) cast their float inputs to the
//! scope's dtype (F16 or BF16), and the losses (`mse_loss`, `cross_entropy_loss`,
//! `nll_loss`, `bce_loss`, `bce_with_logits_loss`, `kl_divergence_gaussian`) cast theirs to
//! F32. Every other op runs in its inputs' dtype, promoting mixed operands as usual: on the
//! CPU, PyTorch keeps softmax, norms and reductions in the half dtype too. The casts are
//! `to_dtype` graph nodes, so parameters stay f32 and receive f32 gradients: there are no
//! master weights to keep, and the optimizer is unchanged. F64 tensors are never cast,
//! as in PyTorch.
//!
//! A `Linear` or convolution is cast as a unit (input, weight and bias), so its bias add
//! stays in the low dtype, as PyTorch's fused `linear` does; casting only the matmul would
//! promote the add back to f32.
//!
//! [`GradScaler`] scales an f16 loss so small gradients don't underflow, unscales the
//! gradients, skips a step whose gradients hold an inf or NaN, and adapts the scale, with
//! PyTorch's defaults. BF16 has f32's exponent range and does not need it.

use std::cell::Cell;
use std::marker::PhantomData;

use crate::dtype::DType;
use crate::nn::optim::Optimizer;
use crate::real::{Real, dispatch_real};

use crate::storage::Storage;
use crate::tensor::Tensor;

thread_local! {
    static AUTOCAST: Cell<Option<DType>> = const { Cell::new(None) };
}

/// The dtype of this thread's innermost [`autocast`] scope, if any.
#[must_use]
pub fn autocast_dtype() -> Option<DType> {
    AUTOCAST.with(Cell::get)
}

/// Run the low-precision ops in `dtype` (F16 or BF16) until the returned guard drops
/// (PyTorch's `torch.autocast("cpu", dtype=..)`). Scopes nest; the guard restores the
/// enclosing scope, on unwind too.
///
/// # Panics
/// If `dtype` is not F16 or BF16.
#[must_use]
#[track_caller]
pub fn autocast(dtype: DType) -> AutocastGuard {
    assert!(
        matches!(dtype, DType::F16 | DType::BF16),
        "autocast: the low-precision dtype must be F16 or BF16, got {dtype}"
    );
    let prev = AUTOCAST.with(|a| a.replace(Some(dtype)));
    AutocastGuard {
        prev,
        _not_send: PhantomData,
    }
}

/// Ends an [`autocast`] scope on drop, restoring the enclosing one.
///
/// `!Send`: the scope is thread-local, like `NoGradGuard`.
pub struct AutocastGuard {
    /// The scope to restore.
    prev: Option<DType>,
    /// Makes the guard `!Send` (and `!Sync`).
    _not_send: PhantomData<*const ()>,
}

impl Drop for AutocastGuard {
    fn drop(&mut self) {
        AUTOCAST.with(|a| a.set(self.prev));
    }
}

/// Whether autocast may cast `t`: a float tensor other than F64 (PyTorch's eligibility).
fn eligible(t: &Tensor) -> bool {
    let d = t.dtype();
    d.is_float() && d != DType::F64
}

/// `t` as a low-precision op's input: cast to the autocast dtype inside a scope, else as is.
pub(crate) fn lower(t: &Tensor) -> Tensor {
    match autocast_dtype() {
        Some(low) if eligible(t) => t.to_dtype(low),
        _ => t.clone(),
    }
}

/// `t` as an f32 op's input (a loss): cast to F32 inside a scope, else as is.
pub(crate) fn to_f32(t: &Tensor) -> Tensor {
    match autocast_dtype() {
        Some(_) if eligible(t) => t.to_dtype(DType::F32),
        _ => t.clone(),
    }
}

/// A layer's input, weight and optional bias, each through [`lower`]: the layer runs as
/// one low-precision op.
pub(crate) fn lower_layer(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
) -> (Tensor, Tensor, Option<Tensor>) {
    (lower(x), lower(weight), bias.map(lower))
}

/// Where a [`GradScaler`] is in its step: torch's per-optimizer stage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stage {
    /// No gradients unscaled since the last `update`.
    Ready,
    /// The gradients are unscaled; `found_inf` says whether any is inf or NaN.
    Unscaled {
        /// An unscaled gradient held an inf or NaN.
        found_inf: bool,
    },
    /// `step` ran (or skipped); waiting for `update`.
    Stepped {
        /// The step was skipped for an inf or NaN gradient.
        found_inf: bool,
    },
}

/// Loss scaling for f16 training (PyTorch's `torch.amp.GradScaler`).
///
/// Each iteration: `scaler.scale(&loss)` before `backward()`, then `scaler.step(&mut opt)`
/// (which unscales the gradients and skips the step if any is inf or NaN), then
/// `scaler.update()`. A skipped step multiplies the scale by `backoff_factor`;
/// `growth_interval` clean steps in a row multiply it by `growth_factor`. Call
/// [`Self::unscale`] first to read or clip the true gradients before `step`.
///
/// ```
/// use volta::amp::{GradScaler, autocast};
/// use volta::{DType, Layer, Linear, Module, RawTensor, SGD};
///
/// volta::manual_seed(0);
/// let model = Linear::new(4, 2, true);
/// let mut opt = SGD::new(model.parameters(), 0.1, 0.0, 0.0);
/// let mut scaler = GradScaler::new();
/// let x = RawTensor::randn(&[8, 4]);
/// let y = RawTensor::randn(&[8, 2]);
/// let mut ran = Vec::new();
/// for _ in 0..8 {
///     let loss = {
///         let _ac = autocast(DType::F16);
///         RawTensor::mse_loss(&model.forward(&x), &y) // f32: losses run in f32
///     };
///     opt.zero_grad();
///     scaler.scale(&loss).into_loss().expect("scalar").backward();
///     ran.push(scaler.step(&mut opt));
///     scaler.update();
/// }
/// // At 2^16 the scaled f16 gradients overflow: those steps are skipped while the scale
/// // halves, then training proceeds (as in PyTorch).
/// assert!(!ran[0] && ran[7]);
/// assert!(scaler.get_scale() < 65536.0);
/// ```
#[derive(Clone, Debug)]
pub struct GradScaler {
    /// The current loss scale.
    scale: f64,
    /// Multiplier after `growth_interval` clean steps.
    growth_factor: f64,
    /// Multiplier after a skipped step.
    backoff_factor: f64,
    /// Clean steps in a row before the scale grows.
    growth_interval: u32,
    /// Clean steps since the last growth or skip.
    growth_tracker: u32,
    /// Where this iteration is.
    stage: Stage,
}

impl Default for GradScaler {
    fn default() -> Self {
        Self::new()
    }
}

impl GradScaler {
    /// PyTorch's defaults: scale 2^16, growth 2, backoff 0.5, growth interval 2000.
    #[must_use]
    pub fn new() -> Self {
        Self::with_params(65536.0, 2.0, 0.5, 2000)
    }

    /// A scaler with explicit settings.
    ///
    /// # Panics
    /// Unless `init_scale > 0`, `growth_factor > 1`, `0 < backoff_factor < 1`, and
    /// `growth_interval > 0` (PyTorch's checks).
    #[must_use]
    #[track_caller]
    pub fn with_params(
        init_scale: f64,
        growth_factor: f64,
        backoff_factor: f64,
        growth_interval: u32,
    ) -> Self {
        assert!(init_scale > 0.0, "GradScaler: init_scale must be > 0");
        assert!(growth_factor > 1.0, "GradScaler: growth_factor must be > 1");
        assert!(
            backoff_factor > 0.0 && backoff_factor < 1.0,
            "GradScaler: backoff_factor must be in (0, 1)"
        );
        assert!(
            growth_interval > 0,
            "GradScaler: growth_interval must be > 0"
        );
        Self {
            scale: init_scale,
            growth_factor,
            backoff_factor,
            growth_interval,
            growth_tracker: 0,
            stage: Stage::Ready,
        }
    }

    /// The current loss scale.
    #[must_use]
    pub fn get_scale(&self) -> f64 {
        self.scale
    }

    /// `loss * scale`, in the loss's dtype and on its device: call `backward()` on this.
    #[must_use]
    pub fn scale(&self, loss: &Tensor) -> Tensor {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the scale is a power of two in practice, exact in f32"
        )]
        let s = self.scale as f32;
        loss.elem_mul(&loss.constant_like(vec![s], &[1]))
    }

    /// Divide every gradient of `opt`'s parameters by the scale, in the gradient's dtype,
    /// and record whether any is inf or NaN. Optional before [`Self::step`] (to clip or
    /// read the true gradients); `step` unscales if it hasn't been done.
    ///
    /// # Panics
    /// If called twice before `update` (torch refuses too: the gradients would be divided
    /// twice), or a gradient is f16 (torch refuses those: loss scaling is for f32
    /// parameters under autocast; bf16 gradients are accepted, as in torch).
    pub fn unscale(&mut self, opt: &impl Optimizer) {
        assert!(
            self.stage == Stage::Ready,
            "GradScaler::unscale: already unscaled since the last update()"
        );
        let inv = 1.0 / self.scale;
        let mut found_inf = false;
        for (i, p) in opt.parameters().iter().enumerate() {
            let Some(g) = p.grad_storage() else { continue };
            // PyTorch's rule: an f16 gradient can under- or overflow in f16 itself, so
            // unscaling it is refused; loss scaling is for autocast's f32 parameters.
            assert!(
                g.dtype() != DType::F16,
                "GradScaler: Attempting to unscale FP16 gradients (parameter {i}); keep the \
                 parameters f32 and run the forward under amp::autocast(DType::F16) (audit A4)"
            );
            let (unscaled, bad) = unscale_storage(&g, inv);
            found_inf |= bad;
            p.set_grad(unscaled);
        }
        self.stage = Stage::Unscaled { found_inf };
    }

    /// Unscale (if not done yet) and run `opt.step()` unless a gradient is inf or NaN.
    /// Returns whether the step ran. Then call [`Self::update`].
    ///
    /// # Panics
    /// If called twice before `update`.
    pub fn step(&mut self, opt: &mut impl Optimizer) -> bool {
        if self.stage == Stage::Ready {
            self.unscale(opt);
        }
        let Stage::Unscaled { found_inf } = self.stage else {
            panic!("GradScaler::step: already stepped since the last update()")
        };
        if !found_inf {
            opt.step();
        }
        self.stage = Stage::Stepped { found_inf };
        !found_inf
    }

    /// Adapt the scale after a step: back off after a skipped step, grow after
    /// `growth_interval` clean ones. Starts the next iteration.
    ///
    /// # Panics
    /// If `step` has not run since the last `update`.
    pub fn update(&mut self) {
        let Stage::Stepped { found_inf } = self.stage else {
            panic!("GradScaler::update: call step() first")
        };
        if found_inf {
            self.scale *= self.backoff_factor;
            self.growth_tracker = 0;
        } else {
            self.growth_tracker += 1;
            if self.growth_tracker == self.growth_interval {
                self.scale *= self.growth_factor;
                self.growth_tracker = 0;
            }
        }
        self.stage = Stage::Ready;
    }
}

/// `g * inv` in `g`'s dtype (computed in f32 for half gradients, rounded once), and
/// whether any result is inf or NaN. A GPU gradient is unscaled on the device by a fused
/// kernel; only the flag comes back.
fn unscale_storage(g: &Storage, inv: f64) -> (Storage, bool) {
    if g.is_gpu() {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the GPU kernels compute in f32"
        )]
        let inv32 = inv as f32;
        return crate::RawTensor::gpu_amp_unscale(g, inv32)
            .unwrap_or_else(|| crate::device_policy::kernel_failed("GradScaler::unscale"));
    }
    let dtype = g.dtype();
    dispatch_real!(dtype, T => {
        let inv = T::from_f64(inv);
        let out: Vec<T> = g.host::<T>().iter().map(|&v| v * inv).collect();
        let s = Storage::from_compute(dtype, out);
        // Checked after rounding: a finite f32 can overflow to inf in f16.
        let bad = s.host::<T>().iter().any(|&v| !v.is_finite());
        (s, bad)
    })
}
