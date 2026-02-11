// src/layers/conv.rs
use crate::Storage;
use crate::autograd::GradFn;
use crate::device::Device;
use crate::io::{StateDict, TensorData};
use crate::nn::Module;
use crate::tensor::{RawTensor, Tensor, TensorOps};

/// Algorithm to use for convolution
#[derive(Default, Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConvAlgo {
    /// Direct convolution - compute each output pixel by iterating over kernel
    /// Memory efficient but slower. Best for small inputs/kernels.
    Direct,
    /// im2col + GEMM - materialize intermediate matrix
    /// Faster for large kernels but high memory usage.
    Im2col,
    /// Implicit GEMM (iGEMM) - performs GEMM without materializing im2col matrix
    /// Good balance of speed and memory. Tiled variant by default.
    IGEMM,
    /// Auto-select based on input characteristics
    #[default]
    Auto,
}

impl ConvAlgo {
    /// Auto-select algorithm based on input characteristics
    #[must_use]
    pub fn auto_select(
        batch: usize,
        channels: usize,
        height: usize,
        width: usize,
        kernel_size: usize,
        has_gpu: bool,
    ) -> Self {
        // Thresholds (tunable)
        const MAX_IM2COL_ELEMENTS: usize = 5_000_000; // ~20 MB for f32

        // Memory heuristic: im2col creates (B*H_out*W_out, C*K*K) matrix
        // Estimate memory and decide
        let h_out = (height - kernel_size) + 1; // Assuming stride=1, pad=0 for heuristic
        let w_out = (width - kernel_size) + 1;
        let im2col_elements = batch * h_out * w_out * channels * kernel_size * kernel_size;

        if has_gpu {
            // On GPU: iGEMM is now GPU-accelerated, providing best balance
            if im2col_elements > 1_000_000 {
                // Medium-to-large inputs - iGEMM avoids im2col memory overhead
                Self::IGEMM
            } else {
                // Small inputs - im2col is fast enough with low memory
                Self::Im2col
            }
        } else if im2col_elements > MAX_IM2COL_ELEMENTS {
            // Large input on CPU - use direct to save memory
            Self::Direct
        } else if kernel_size <= 3 && batch <= 4 {
            // Small kernel and batch on CPU - direct is fast enough and memory efficient
            Self::Direct
        } else if im2col_elements > 1_000_000 {
            // Medium-to-large inputs on CPU - iGEMM provides good balance
            // Avoids im2col memory overhead while being faster than direct
            Self::IGEMM
        } else {
            // Small-to-medium inputs where im2col is acceptable
            Self::Im2col
        }
    }
}

/// Implicit GEMM (iGEMM) variants for convolution
///
/// This enum defines different iGEMM implementations that avoid materializing
/// the im2col matrix while leveraging GEMM-style computation.
///
/// # Extensibility
///
/// To add a new iGEMM variant:
/// 1. Add a new variant to this enum
/// 2. Implement the forward/backward computation in `Conv2d::igemm_forward`
/// 3. Update `Conv2d::igemm_backward` to handle the new variant
/// 4. Add tests and benchmark comparisons
///
/// # Variant Selection Guide
///
/// - **Tiled**: Good all-rounder, works well for most cases. Cache-friendly blocking.
/// - **`Winograd`** (future): Fastest for 3x3 kernels with stride 1. Transform-domain approach.
/// - **`DirectToGEMM`** (future): Best performance with hand-tuned micro-kernels.
/// - **`FFT`** (future): Best for large kernels (audio processing). O(n log n) complexity.
#[derive(Default, Clone, Copy, Debug, PartialEq, Eq)]
enum IGEMMVariant {
    /// Tiled iGEMM - blocks computation for cache efficiency
    /// Good balance of performance and memory usage
    #[default]
    Tiled,
    // Future variants (extension points):
    // Winograd { transform_size: usize },  // e.g., Winograd(2x2) or Winograd(4x4)
    // DirectToGEMM,                       // Custom micro-kernel approach
    // FFT,                                // Frequency-domain convolution
}

/// Configuration for tiled iGEMM computation
///
/// These parameters control the tiling strategy for cache optimization.
/// Different hardware may benefit from different tile sizes.
#[derive(Clone, Copy, Debug)]
struct TileConfig {
    /// Tile size for output channels dimension (O)
    pub output_channels: usize,
    /// Tile size for input channels dimension (C)
    pub input_channels: usize,
    /// Tile size for output spatial dimensions `(H_out * W_out)`
    pub output_spacial_dims: usize,
}

impl Default for TileConfig {
    fn default() -> Self {
        // Reasonable defaults for typical CPU cache sizes
        // L1 cache is usually 32-64KB, L2 is 256-512KB
        Self {
            output_channels: 32,     // 32 output channels ~ 4KB for f32
            input_channels: 16,      // 16 input channels
            output_spacial_dims: 64, // 64 output positions
        }
    }
}

pub struct Conv2d {
    weight: Tensor,       // [out_channels, in_channels, kernel_h, kernel_w]
    bias: Option<Tensor>, // [out_channels]
    stride: (usize, usize),
    padding: (usize, usize),
    algo: std::cell::Cell<ConvAlgo>, // Algorithm to use
}

/// Gradient function for im2col operation
#[derive(Clone)]
struct Im2colGradFn {
    input_shape: Vec<usize>,
    kernel: (usize, usize),
    stride: (usize, usize),
}

impl GradFn for Im2colGradFn {
    fn backward(&self, out_grad: &RawTensor, _parents: &[Tensor]) -> Vec<Option<Tensor>> {
        // out_grad has shape (B*H_out*W_out, C*K*K)
        // We need to convert it back to (B, C, H, W) using col2im

        // Try GPU-accelerated col2im first
        let grad_storage = if out_grad.data.is_gpu() {
            let (batch, channels, height, width) = (
                self.input_shape.first().copied().unwrap_or(1),
                self.input_shape.get(1).copied().unwrap_or(1),
                self.input_shape.get(2).copied().unwrap_or(1),
                self.input_shape.get(3).copied().unwrap_or(1),
            );
            let (kh, kw) = self.kernel;
            let (sh, sw) = self.stride;
            let h_out = (height - kh) / sh + 1;
            let w_out = (width - kw) / sw + 1;

            // Try GPU col2im
            if let Some(gpu_storage) = RawTensor::gpu_col2im(
                &out_grad.data,
                batch,
                channels,
                height,
                width,
                kh,
                kw,
                sh,
                sw,
                h_out,
                w_out,
            ) {
                gpu_storage
            } else {
                // Fallback to CPU - convert Vec<f32> to Storage
                let cpu_data =
                    Conv2d::col2im(&out_grad.data, &self.input_shape, self.kernel, self.stride);
                crate::storage::Storage::cpu(cpu_data)
            }
        } else {
            // CPU path - convert Vec<f32> to Storage
            let cpu_data =
                Conv2d::col2im(&out_grad.data, &self.input_shape, self.kernel, self.stride);
            crate::storage::Storage::cpu(cpu_data)
        };

        // Determine device from storage
        let device = if grad_storage.is_gpu() {
            crate::Device::gpu().unwrap_or(crate::Device::CPU)
        } else {
            crate::Device::CPU
        };

        let grad_tensor =
            RawTensor::try_new_with_storage(grad_storage, &self.input_shape, device, false)
                .expect("col2im gradient shape mismatch");

        vec![Some(grad_tensor)]
    }

    fn clone_box(&self) -> Box<dyn GradFn> {
        Box::new(self.clone())
    }
}

/// Gradient function for direct convolution operation
///
/// Computes gradients for direct convolution:
/// - Input gradient: Reverse convolution (accumulate from output positions)
/// - Weight gradient: Direct accumulation (outer products of input patches and output gradients)
#[derive(Clone)]
struct DirectConvGradFn {
    input_shape: Vec<usize>,  // (B, C, H, W)
    weight_shape: Vec<usize>, // (O, C, K, K)
    padding: (usize, usize),  // (pad_h, pad_w)
    stride: (usize, usize),   // (stride_h, stride_w)
}

impl GradFn for DirectConvGradFn {
    fn backward(&self, out_grad: &RawTensor, parents: &[Tensor]) -> Vec<Option<Tensor>> {
        // parents: [input, weight]
        let x = parents.first().cloned();
        let weight = parents.get(1).cloned();

        // Unpack shapes
        let (batch, in_ch, height, width) = (
            self.input_shape.first().copied().unwrap_or(1),
            self.input_shape.get(1).copied().unwrap_or(1),
            self.input_shape.get(2).copied().unwrap_or(1),
            self.input_shape.get(3).copied().unwrap_or(1),
        );

        let (out_ch, _, kernel_h, kernel_w) = (
            self.weight_shape.first().copied().unwrap_or(1),
            self.weight_shape.get(1).copied().unwrap_or(1),
            self.weight_shape.get(2).copied().unwrap_or(1),
            self.weight_shape.get(3).copied().unwrap_or(1),
        );

        let (pad_h, pad_w) = self.padding;
        let (stride_h, stride_w) = self.stride;

        let padded_h = height + 2 * pad_h;
        let padded_w = width + 2 * pad_w;
        let h_out = (padded_h - kernel_h) / stride_h + 1;
        let w_out = (padded_w - kernel_w) / stride_w + 1;

        // Determine if we can use GPU acceleration
        #[cfg(feature = "gpu")]
        let use_gpu = matches!(&out_grad.data, Storage::Gpu { .. })
            && x.as_ref().is_some_and(|t| t.borrow().data.is_gpu())
            && weight.as_ref().is_some_and(|t| t.borrow().data.is_gpu());

        #[cfg(not(feature = "gpu"))]
        let use_gpu = false;

        // Compute input gradient if input requires grad
        let grad_x = if let Some(ref x_tensor) = x
            && x_tensor.borrow().requires_grad
        {
            #[cfg(feature = "gpu")]
            if use_gpu {
                // GPU acceleration
                RawTensor::gpu_conv_backward_input(
                    &out_grad.data,
                    &weight.as_ref().unwrap().borrow().data,
                    batch,
                    in_ch,
                    out_ch,
                    height,
                    width,
                    kernel_h,
                    kernel_w,
                    stride_h,
                    stride_w,
                    pad_h,
                    pad_w,
                    h_out,
                    w_out,
                )
                .map(|storage| {
                    RawTensor::new_with_storage(
                        storage,
                        &self.input_shape,
                        out_grad.device.clone(),
                        false,
                    )
                })
            } else {
                // CPU fallback
                Some(self.compute_input_gradient_cpu(
                    &out_grad.data.to_vec(),
                    &weight.as_ref().unwrap().borrow().data.to_vec(),
                    batch,
                    in_ch,
                    out_ch,
                    height,
                    width,
                    kernel_h,
                    kernel_w,
                    stride_h,
                    stride_w,
                    pad_h,
                    pad_w,
                    h_out,
                    w_out,
                ))
            }

            #[cfg(not(feature = "gpu"))]
            {
                self.compute_input_gradient_cpu(
                    &out_grad.data.to_vec(),
                    &weight.as_ref().unwrap().borrow().data.to_vec(),
                    batch,
                    in_ch,
                    out_ch,
                    height,
                    width,
                    kernel_h,
                    kernel_w,
                    stride_h,
                    stride_w,
                    pad_h,
                    pad_w,
                    h_out,
                    w_out,
                )
            }
        } else {
            None
        };

        // Compute weight gradient if weight requires grad
        let grad_w = if let Some(ref w_tensor) = weight
            && w_tensor.borrow().requires_grad
        {
            #[cfg(feature = "gpu")]
            if use_gpu {
                // GPU acceleration
                RawTensor::gpu_conv_backward_weight(
                    &out_grad.data,
                    &x.as_ref().unwrap().borrow().data,
                    batch,
                    in_ch,
                    out_ch,
                    height,
                    width,
                    kernel_h,
                    kernel_w,
                    stride_h,
                    stride_w,
                    pad_h,
                    pad_w,
                    h_out,
                    w_out,
                )
                .map(|storage| {
                    RawTensor::new_with_storage(
                        storage,
                        &self.weight_shape,
                        out_grad.device.clone(),
                        false,
                    )
                })
            } else {
                // CPU fallback
                Some(self.compute_weight_gradient_cpu(
                    &out_grad.data.to_vec(),
                    &x.as_ref().unwrap().borrow().data.to_vec(),
                    batch,
                    in_ch,
                    out_ch,
                    height,
                    width,
                    kernel_h,
                    kernel_w,
                    stride_h,
                    stride_w,
                    pad_h,
                    pad_w,
                    h_out,
                    w_out,
                ))
            }

            #[cfg(not(feature = "gpu"))]
            {
                self.compute_weight_gradient_cpu(
                    &out_grad.data.to_vec(),
                    &x.as_ref().unwrap().borrow().data.to_vec(),
                    batch,
                    in_ch,
                    out_ch,
                    height,
                    width,
                    kernel_h,
                    kernel_w,
                    stride_h,
                    stride_w,
                    pad_h,
                    pad_w,
                    h_out,
                    w_out,
                )
            }
        } else {
            None
        };

        vec![grad_x, grad_w]
    }

    fn clone_box(&self) -> Box<dyn GradFn> {
        Box::new(self.clone())
    }
}

impl DirectConvGradFn {
    /// CPU implementation of input gradient computation
    #[allow(clippy::too_many_arguments)]
    fn compute_input_gradient_cpu(
        &self,
        out_grad_data: &[f32],
        weight_data: &[f32],
        batch: usize,
        in_ch: usize,
        out_ch: usize,
        height: usize,
        width: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
        pad_h: usize,
        pad_w: usize,
        h_out: usize,
        w_out: usize,
    ) -> Tensor {
        let mut grad_x_data = vec![0.0f32; batch * in_ch * height * width];

        // Input gradient: full convolution of output gradient with flipped weights
        // For each input position, accumulate contributions from all output positions
        // that use this input value with the corresponding (flipped) weight
        for b in 0..batch {
            for ic in 0..in_ch {
                for ih in 0..height {
                    for iw in 0..width {
                        let mut sum = 0.0f32;

                        // Iterate over all output positions and kernel positions
                        // to find which ones contribute to this input position
                        for oc in 0..out_ch {
                            for oh in 0..h_out {
                                for ow in 0..w_out {
                                    // Get output gradient
                                    let out_idx = b * (out_ch * h_out * w_out)
                                        + oc * (h_out * w_out)
                                        + oh * w_out
                                        + ow;
                                    let dout = if let Some(&val) = out_grad_data.get(out_idx) {
                                        val
                                    } else {
                                        0.0
                                    };

                                    // Starting position in input for this output position
                                    let h_start = oh * stride_h;
                                    let w_start = ow * stride_w;

                                    // Check if this input position (ih, iw) is in the receptive field
                                    // of output position (oh, ow)
                                    let h_offset = ih as isize - h_start as isize + pad_h as isize;
                                    let w_offset = iw as isize - w_start as isize + pad_w as isize;

                                    if h_offset >= 0
                                        && w_offset >= 0
                                        && h_offset < kernel_h as isize
                                        && w_offset < kernel_w as isize
                                    {
                                        let kh = h_offset as usize;
                                        let kw = w_offset as usize;

                                        // Get the corresponding weight value
                                        // Note: weights are indexed as (oc, ic, kh, kw)
                                        let w_idx = oc * (in_ch * kernel_h * kernel_w)
                                            + ic * (kernel_h * kernel_w)
                                            + kh * kernel_w
                                            + kw;
                                        let w_val = if let Some(&val) = weight_data.get(w_idx) {
                                            val
                                        } else {
                                            0.0
                                        };

                                        sum += dout * w_val;
                                    }
                                }
                            }
                        }

                        let in_idx =
                            b * (in_ch * height * width) + ic * (height * width) + ih * width + iw;
                        if let Some(slot) = grad_x_data.get_mut(in_idx) {
                            *slot = sum;
                        }
                    }
                }
            }
        }

        RawTensor::new(grad_x_data, &self.input_shape, false)
    }

    /// CPU implementation of weight gradient computation
    #[allow(clippy::too_many_arguments)]
    fn compute_weight_gradient_cpu(
        &self,
        out_grad_data: &[f32],
        x_data: &[f32],
        batch: usize,
        in_ch: usize,
        out_ch: usize,
        height: usize,
        width: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
        pad_h: usize,
        pad_w: usize,
        h_out: usize,
        w_out: usize,
    ) -> Tensor {
        let mut grad_w_data = vec![0.0f32; out_ch * in_ch * kernel_h * kernel_w];

        // Weight gradient: sum of outer products of input patches and output gradients
        // ∂L/∂W[o,ic,kh,kw] = sum_{b,oh,ow} X[b,ic,oh*sh+kh-ph,ow*sw+kw-pw] * ∂L/∂Y[b,o,oh,ow]
        for b in 0..batch {
            for oc in 0..out_ch {
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        // Get output gradient
                        let out_idx =
                            b * (out_ch * h_out * w_out) + oc * (h_out * w_out) + oh * w_out + ow;
                        let dout = if let Some(&val) = out_grad_data.get(out_idx) {
                            val
                        } else {
                            0.0
                        };

                        // Starting position in input (accounting for padding)
                        let h_start = oh * stride_h;
                        let w_start = ow * stride_w;

                        // Accumulate gradient for each weight element
                        for ic in 0..in_ch {
                            for kh in 0..kernel_h {
                                for kw in 0..kernel_w {
                                    let h_pos = h_start + kh;
                                    let w_pos = w_start + kw;

                                    // Check if within padded bounds
                                    let in_h = h_pos as isize - pad_h as isize;
                                    let in_w = w_pos as isize - pad_w as isize;

                                    if in_h >= 0
                                        && in_h < height as isize
                                        && in_w >= 0
                                        && in_w < width as isize
                                    {
                                        let in_h = in_h as usize;
                                        let in_w = in_w as usize;

                                        // Get input value
                                        let in_idx = b * (in_ch * height * width)
                                            + ic * (height * width)
                                            + in_h * width
                                            + in_w;
                                        let in_val = if let Some(&val) = x_data.get(in_idx) {
                                            val
                                        } else {
                                            0.0
                                        };

                                        // Accumulate weight gradient
                                        let w_idx = oc * (in_ch * kernel_h * kernel_w)
                                            + ic * (kernel_h * kernel_w)
                                            + kh * kernel_w
                                            + kw;
                                        if let Some(slot) = grad_w_data.get_mut(w_idx) {
                                            *slot += in_val * dout;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        RawTensor::new(grad_w_data, &self.weight_shape, false)
    }
}

/// Gradient function for implicit GEMM convolution
///
/// Computes gradients for iGEMM operation using the same tiling strategy
/// as the forward pass to maintain consistency and efficiency.
#[derive(Clone)]
struct IGEMMGradFn {
    input_shape: Vec<usize>,  // (B, C, H, W)
    weight_shape: Vec<usize>, // (O, C, K, K)
    padding: (usize, usize),  // (pad_h, pad_w)
    stride: (usize, usize),   // (stride_h, stride_w)
    _variant: IGEMMVariant,   // Which iGEMM variant was used
}

impl GradFn for IGEMMGradFn {
    fn backward(&self, out_grad: &RawTensor, parents: &[Tensor]) -> Vec<Option<Tensor>> {
        // parents: [input, weight]
        let x = parents.first().cloned();
        let weight = parents.get(1).cloned();

        // Unpack shapes
        let (batch, in_ch, height, width) = (
            self.input_shape.first().copied().unwrap_or(1),
            self.input_shape.get(1).copied().unwrap_or(1),
            self.input_shape.get(2).copied().unwrap_or(1),
            self.input_shape.get(3).copied().unwrap_or(1),
        );

        let (out_ch, _, kernel_h, kernel_w) = (
            self.weight_shape.first().copied().unwrap_or(1),
            self.weight_shape.get(1).copied().unwrap_or(1),
            self.weight_shape.get(2).copied().unwrap_or(1),
            self.weight_shape.get(3).copied().unwrap_or(1),
        );

        let (pad_h, pad_w) = self.padding;
        let (stride_h, stride_w) = self.stride;

        let padded_h = height + 2 * pad_h;
        let padded_w = width + 2 * pad_w;
        let h_out = (padded_h - kernel_h) / stride_h + 1;
        let w_out = (padded_w - kernel_w) / stride_w + 1;

        // Check if we're on GPU
        let is_gpu = matches!(out_grad.device, Device::GPU(_));

        // Compute input gradient
        let grad_x = if let Some(ref x_tensor) = x
            && x_tensor.borrow().requires_grad
        {
            #[cfg(feature = "gpu")]
            let result = if is_gpu {
                RawTensor::gpu_igemm_backward_input(
                    &out_grad.data,
                    &weight.as_ref().unwrap().borrow().data,
                    batch,
                    in_ch,
                    out_ch,
                    height,
                    width,
                    kernel_h,
                    kernel_w,
                    stride_h,
                    stride_w,
                    pad_h,
                    pad_w,
                    h_out,
                    w_out,
                )
                .map(|storage| {
                    RawTensor::new_with_storage(
                        storage,
                        &self.input_shape,
                        out_grad.device.clone(),
                        false,
                    )
                })
            } else {
                None
            };

            #[cfg(not(feature = "gpu"))]
            let result = None;

            if let Some(r) = result {
                Some(r)
            } else {
                // CPU fallback
                let mut grad_x_data = vec![0.0f32; batch * in_ch * height * width];
                let out_grad_data = out_grad.data.to_vec();
                let weight_data = weight.as_ref().map(|w| w.borrow().data.to_vec());

                if let Some(ref wd) = weight_data {
                    // Input gradient: full convolution of output grad with flipped weights
                    // Use same tiling as forward pass
                    let config = TileConfig::default();
                    Self::igemm_backward_input(
                        &out_grad_data,
                        wd,
                        &mut grad_x_data,
                        batch,
                        in_ch,
                        out_ch,
                        height,
                        width,
                        kernel_h,
                        kernel_w,
                        stride_h,
                        stride_w,
                        pad_h,
                        pad_w,
                        h_out,
                        w_out,
                        &config,
                    );
                }

                Some(RawTensor::new(grad_x_data, &self.input_shape, false))
            }
        } else {
            None
        };

        // Compute weight gradient
        let grad_w = if let Some(ref w_tensor) = weight
            && w_tensor.borrow().requires_grad
        {
            #[cfg(feature = "gpu")]
            let result = if is_gpu {
                RawTensor::gpu_igemm_backward_weight(
                    &out_grad.data,
                    &x.as_ref().unwrap().borrow().data,
                    batch,
                    in_ch,
                    out_ch,
                    height,
                    width,
                    kernel_h,
                    kernel_w,
                    stride_h,
                    stride_w,
                    pad_h,
                    pad_w,
                    h_out,
                    w_out,
                )
                .map(|storage| {
                    RawTensor::new_with_storage(
                        storage,
                        &self.weight_shape,
                        out_grad.device.clone(),
                        false,
                    )
                })
            } else {
                None
            };

            #[cfg(not(feature = "gpu"))]
            let result = None;

            if let Some(r) = result {
                Some(r)
            } else {
                // CPU fallback
                let mut grad_w_data = vec![0.0f32; out_ch * in_ch * kernel_h * kernel_w];
                let out_grad_data = out_grad.data.to_vec();
                let x_data = x.as_ref().map(|x| x.borrow().data.to_vec());

                if let Some(ref xd) = x_data {
                    // Weight gradient: sum over outer products
                    let config = TileConfig::default();
                    Self::igemm_backward_weight(
                        xd,
                        &out_grad_data,
                        &mut grad_w_data,
                        batch,
                        in_ch,
                        out_ch,
                        height,
                        width,
                        kernel_h,
                        kernel_w,
                        stride_h,
                        stride_w,
                        pad_h,
                        pad_w,
                        h_out,
                        w_out,
                        &config,
                    );
                }

                Some(RawTensor::new(grad_w_data, &self.weight_shape, false))
            }
        } else {
            None
        };

        vec![grad_x, grad_w]
    }

    fn clone_box(&self) -> Box<dyn GradFn> {
        Box::new(self.clone())
    }
}

impl IGEMMGradFn {
    /// Compute input gradient for iGEMM using tiled approach
    #[allow(clippy::too_many_arguments)]
    fn igemm_backward_input(
        out_grad: &[f32],
        weight: &[f32],
        grad_x: &mut [f32],
        batch: usize,
        in_ch: usize,
        out_ch: usize,
        height: usize,
        width: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
        pad_h: usize,
        pad_w: usize,
        h_out: usize,
        w_out: usize,
        config: &TileConfig,
    ) {
        let kernel_size = kernel_h * kernel_w;
        let spatial_out = h_out * w_out;

        // Tile over input channels and spatial positions
        for ic_start in (0..in_ch).step_by(config.input_channels) {
            let ic_end = (ic_start + config.input_channels).min(in_ch);

            for s_start in (0..batch * height * width).step_by(config.output_spacial_dims) {
                let s_end = (s_start + config.output_spacial_dims).min(batch * height * width);

                // For each input position in tile:
                for s_idx in s_start..s_end {
                    let b = s_idx / (height * width);
                    let remaining = s_idx % (height * width);
                    let ih = remaining / width;
                    let iw = remaining % width;

                    // For each input channel in tile:
                    for ic in ic_start..ic_end {
                        let mut sum = 0.0;

                        // Accumulate from all output positions that use this input
                        for oc in 0..out_ch {
                            for oh in 0..h_out {
                                for ow in 0..w_out {
                                    // Check if this input position is in the receptive field
                                    let h_start = oh * stride_h;
                                    let w_start = ow * stride_w;

                                    let h_offset = ih as isize - h_start as isize + pad_h as isize;
                                    let w_offset = iw as isize - w_start as isize + pad_w as isize;

                                    if h_offset >= 0
                                        && h_offset < kernel_h as isize
                                        && w_offset >= 0
                                        && w_offset < kernel_w as isize
                                    {
                                        let kh = h_offset as usize;
                                        let kw = w_offset as usize;

                                        // Output grad index: (b, oc, oh, ow)
                                        let out_idx = b * (out_ch * spatial_out)
                                            + oc * spatial_out
                                            + oh * w_out
                                            + ow;

                                        // Weight index: (oc, ic, kh, kw)
                                        let w_idx = oc * (in_ch * kernel_size)
                                            + ic * kernel_size
                                            + kh * kernel_w
                                            + kw;

                                        if let (Some(&dout), Some(&w_val)) =
                                            (out_grad.get(out_idx), weight.get(w_idx))
                                        {
                                            sum += dout * w_val;
                                        }
                                    }
                                }
                            }
                        }

                        // Input grad index: (b, ic, ih, iw)
                        let in_idx = b * (in_ch * height * width) + ic * (height * width) + s_idx;
                        if let Some(slot) = grad_x.get_mut(in_idx) {
                            *slot = sum;
                        }
                    }
                }
            }
        }
    }

    /// Compute weight gradient for iGEMM using tiled approach
    #[allow(clippy::too_many_arguments)]
    fn igemm_backward_weight(
        x: &[f32],
        out_grad: &[f32],
        grad_w: &mut [f32],
        batch: usize,
        in_ch: usize,
        out_ch: usize,
        height: usize,
        width: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
        pad_h: usize,
        pad_w: usize,
        h_out: usize,
        w_out: usize,
        config: &TileConfig,
    ) {
        let kernel_size = kernel_h * kernel_w;
        let spatial_out = h_out * w_out;

        // Tile over output channels and spatial positions
        for oc_start in (0..out_ch).step_by(config.output_channels) {
            let oc_end = (oc_start + config.output_channels).min(out_ch);

            for s_start in (0..batch * spatial_out).step_by(config.output_spacial_dims) {
                let s_end = (s_start + config.output_spacial_dims).min(batch * spatial_out);

                // Tile over input channels
                for ic_start in (0..in_ch).step_by(config.input_channels) {
                    let ic_end = (ic_start + config.input_channels).min(in_ch);

                    // For each output position in tile:
                    for s_idx in s_start..s_end {
                        let b = s_idx / spatial_out;
                        let remaining = s_idx % spatial_out;
                        let oh = remaining / w_out;
                        let ow = remaining % w_out;

                        // Starting position in input
                        let h_start = oh * stride_h;
                        let w_start = ow * stride_w;

                        // Output grad index: (b, oc, oh, ow)
                        let out_idx_base = b * (out_ch * spatial_out) + s_idx;

                        // For each output channel and kernel position:
                        for oc in oc_start..oc_end {
                            let out_idx = out_idx_base + oc * spatial_out;
                            let dout = *out_grad.get(out_idx).unwrap_or(&0.0);

                            for ic in ic_start..ic_end {
                                for kh in 0..kernel_h {
                                    for kw in 0..kernel_w {
                                        let h_pos = h_start + kh;
                                        let w_pos = w_start + kw;

                                        // Check padding
                                        let in_h = h_pos as isize - pad_h as isize;
                                        let in_w = w_pos as isize - pad_w as isize;

                                        if in_h >= 0
                                            && in_h < height as isize
                                            && in_w >= 0
                                            && in_w < width as isize
                                        {
                                            let in_h = in_h as usize;
                                            let in_w = in_w as usize;

                                            // Input index: (b, ic, in_h, in_w)
                                            let in_idx = b * (in_ch * height * width)
                                                + ic * (height * width)
                                                + in_h * width
                                                + in_w;

                                            // Weight grad index: (oc, ic, kh, kw)
                                            let w_idx = oc * (in_ch * kernel_size)
                                                + ic * kernel_size
                                                + kh * kernel_w
                                                + kw;

                                            if let (Some(&in_val), Some(slot)) =
                                                (x.get(in_idx), grad_w.get_mut(w_idx))
                                            {
                                                *slot += in_val * dout;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

impl Conv2d {
    #[must_use]
    pub fn new(
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
        use_bias: bool,
    ) -> Self {
        let w = RawTensor::he_initialization(&[out_ch, in_ch, kernel, kernel]);
        w.borrow_mut().requires_grad = true;
        let b = if use_bias {
            let b = RawTensor::zeros(&[out_ch]);
            b.borrow_mut().requires_grad = true;
            Some(b)
        } else {
            None
        };
        Self {
            weight: w,
            bias: b,
            stride: (stride, stride),
            padding: (padding, padding),
            algo: std::cell::Cell::new(ConvAlgo::Auto),
        }
    }

    /// Create a new Conv2d layer on a specific device
    ///
    /// Uses He initialization and places tensors on the specified device.
    ///
    /// # Arguments
    /// * `in_ch` - Number of input channels
    /// * `out_ch` - Number of output channels
    /// * `kernel` - Kernel size (square)
    /// * `stride` - Stride (square)
    /// * `padding` - Padding (square)
    /// * `use_bias` - Whether to include a bias term
    /// * `device` - Device to place parameters on (CPU or GPU)
    ///
    /// # Example
    /// ```no_run
    /// # use volta::{Conv2d, Device};
    /// # #[cfg(feature = "gpu")]
    /// # {
    /// let device = Device::gpu().expect("GPU required");
    /// let layer = Conv2d::new_on_device(3, 64, 3, 1, 1, true, device);
    /// // Parameters are now on GPU
    /// # }
    /// ```
    #[must_use]
    pub fn new_on_device(
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
        use_bias: bool,
        device: Device,
    ) -> Self {
        let w = RawTensor::he_initialization(&[out_ch, in_ch, kernel, kernel]);
        w.borrow_mut().requires_grad = true;
        let w = w.to_device(device.clone());

        let b = if use_bias {
            let b = RawTensor::zeros(&[out_ch]);
            b.borrow_mut().requires_grad = true;
            Some(b.to_device(device))
        } else {
            None
        };
        Self {
            weight: w,
            bias: b,
            stride: (stride, stride),
            padding: (padding, padding),
            algo: std::cell::Cell::new(ConvAlgo::Auto),
        }
    }

    /// Set the convolution algorithm to use
    ///
    /// # Arguments
    /// * `algo` - Algorithm to use (Direct, Im2col, or Auto)
    ///
    /// # Example
    /// ```no_run
    /// # use volta::Conv2d;
    /// # use volta::nn::layers::conv::ConvAlgo;
    /// let conv = Conv2d::new(3, 16, 3, 1, 1, true);
    /// conv.set_algo(ConvAlgo::Direct);  // Force direct convolution
    /// ```
    pub fn set_algo(&self, algo: ConvAlgo) {
        self.algo.set(algo);
    }

    /// Get the current convolution algorithm setting
    pub fn get_algo(&self) -> ConvAlgo {
        self.algo.get()
    }

    /// Im2col: Convert (B, C, H, W) → (B*`H_out`*`W_out`, C*K*K) matrix
    /// Each row contains a flattened receptive field
    fn im2col(x: &Tensor, kernel: (usize, usize), stride: (usize, usize)) -> Tensor {
        const MAX_ALLOC: usize = 100_000_000; // Maximum allowed allocation

        let (data, shape, requires_grad, device) = {
            let x_borrow = x.borrow();
            assert_eq!(x_borrow.shape.len(), 4, "Input must be 4D: (B, C, H, W)");
            (
                x_borrow.data.clone(),
                x_borrow.shape.clone(),
                x_borrow.requires_grad,
                x_borrow.device.clone(),
            )
        };

        let (batch, channels, height, width) = (
            shape.first().copied().unwrap_or(1),
            shape.get(1).copied().unwrap_or(1),
            shape.get(2).copied().unwrap_or(1),
            shape.get(3).copied().unwrap_or(1),
        );
        let (kh, kw) = kernel;
        let (sh, sw) = stride;

        // Check for reasonable parameters to prevent memory issues
        assert!(height >= kh && width >= kw, "Input smaller than kernel");
        assert!(
            kh > 0 && kw > 0 && sh > 0 && sw > 0,
            "Invalid kernel/stride parameters"
        );

        // Calculate output dimensions
        let h_out = (height - kh) / sh + 1;
        let w_out = (width - kw) / sw + 1;

        // Output shape: (B*H_out*W_out, C*K*K)
        assert!(h_out > 0 && w_out > 0, "Invalid output dimensions");
        let rows = batch * h_out * w_out;
        let cols = channels * kh * kw;

        // Use GPU path if available and input is on GPU
        #[cfg(feature = "gpu")]
        {
            if matches!(device, Device::GPU(_))
                && let Some(col_data) = crate::RawTensor::gpu_im2col(
                    &data, batch, channels, height, width, kh, kw, sh, sw, h_out, w_out,
                )
            {
                let out =
                    RawTensor::new_with_storage(col_data, &[rows, cols], device, requires_grad);
                if requires_grad {
                    out.borrow_mut().parents = vec![x.clone()];
                    out.borrow_mut().grad_fn = Some(Box::new(Im2colGradFn {
                        input_shape: shape,
                        kernel,
                        stride,
                    }));
                }
                return out;
            }
        }

        // CPU fallback
        let total_elements = rows * cols;
        assert!(
            total_elements <= MAX_ALLOC,
            "im2col would create tensor with {total_elements} elements (max: {MAX_ALLOC}). Input shape: {shape:?}, kernel: {kernel:?}, stride: {stride:?}"
        );

        let mut result = vec![0.0; total_elements];

        // For each output position, extract and flatten the receptive field
        for b in 0..batch {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let row_idx = b * (h_out * w_out) + oh * w_out + ow;

                    // Starting position in input
                    let h_start = oh * sh;
                    let w_start = ow * sw;

                    // Extract receptive field
                    for c in 0..channels {
                        for kh_idx in 0..kh {
                            for kw_idx in 0..kw {
                                let h_pos = h_start + kh_idx;
                                let w_pos = w_start + kw_idx;

                                let in_idx = b * (channels * height * width)
                                    + c * (height * width)
                                    + h_pos * width
                                    + w_pos;

                                let col_idx = c * (kh * kw) + kh_idx * kw + kw_idx;
                                let result_idx = row_idx * cols + col_idx;
                                let cpu_data = data.to_vec();
                                if let Some(&src_val) = cpu_data.get(in_idx)
                                    && let Some(slot) = result.get_mut(result_idx)
                                {
                                    *slot = src_val;
                                }
                            }
                        }
                    }
                }
            }
        }

        let out = RawTensor::new(result, &[rows, cols], requires_grad);

        // Attach gradient function if input requires gradients
        if requires_grad {
            out.borrow_mut().parents = vec![x.clone()];
            out.borrow_mut().grad_fn = Some(Box::new(Im2colGradFn {
                input_shape: shape,
                kernel,
                stride,
            }));
        }

        out
    }

    /// Col2im: Inverse of im2col, used for computing input gradients
    fn col2im(
        col: &[f32],
        output_shape: &[usize], // (B, C, H, W)
        kernel: (usize, usize),
        stride: (usize, usize),
    ) -> Vec<f32> {
        let (batch, channels, height, width) = (
            output_shape.first().copied().unwrap_or(1),
            output_shape.get(1).copied().unwrap_or(1),
            output_shape.get(2).copied().unwrap_or(1),
            output_shape.get(3).copied().unwrap_or(1),
        );
        let (kh, kw) = kernel;
        let (sh, sw) = stride;

        let h_out = (height - kh) / sh + 1;
        let w_out = (width - kw) / sw + 1;

        let mut result = vec![0.0; batch * channels * height * width];

        // Accumulate gradients from each receptive field
        for b in 0..batch {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let row_idx = b * (h_out * w_out) + oh * w_out + ow;

                    let h_start = oh * sh;
                    let w_start = ow * sw;

                    for c in 0..channels {
                        for kh_idx in 0..kh {
                            for kw_idx in 0..kw {
                                let h_pos = h_start + kh_idx;
                                let w_pos = w_start + kw_idx;

                                let out_idx = b * (channels * height * width)
                                    + c * (height * width)
                                    + h_pos * width
                                    + w_pos;

                                let col_idx = c * (kh * kw) + kh_idx * kw + kw_idx;
                                let cols = channels * kh * kw;
                                let col_data_idx = row_idx * cols + col_idx;
                                if let Some(&col_val) = col.get(col_data_idx)
                                    && let Some(slot) = result.get_mut(out_idx)
                                {
                                    *slot += col_val;
                                }
                            }
                        }
                    }
                }
            }
        }

        result
    }

    /// Direct convolution: compute each output pixel by iterating over the kernel
    /// Memory efficient (no intermediate allocation) but slower than im2col+GEMM
    ///
    /// Input: `(B, C, H, W)`, Weight: `(O, C, K, K)`
    /// Output: `(B, O, H_out, W_out)`
    fn direct_conv(
        x: &Tensor,
        weight: &Tensor,
        pad_h: usize,
        pad_w: usize,
        stride_h: usize,
        stride_w: usize,
    ) -> Tensor {
        let (x_data, x_shape, _x_device, x_requires_grad) = {
            let x_borrow = x.borrow();
            (
                x_borrow.data.clone(),
                x_borrow.shape.clone(),
                x_borrow.device.clone(),
                x_borrow.requires_grad,
            )
        };

        let (w_data, w_shape) = {
            let w_borrow = weight.borrow();
            (w_borrow.data.clone(), w_borrow.shape.clone())
        };

        let (batch, in_ch, height, width) = (
            x_shape.first().copied().unwrap_or(1),
            x_shape.get(1).copied().unwrap_or(1),
            x_shape.get(2).copied().unwrap_or(1),
            x_shape.get(3).copied().unwrap_or(1),
        );

        let (out_ch, _, kernel_h, kernel_w) = (
            w_shape.first().copied().unwrap_or(1),
            w_shape.get(1).copied().unwrap_or(1),
            w_shape.get(2).copied().unwrap_or(1),
            w_shape.get(3).copied().unwrap_or(1),
        );

        // Apply padding
        let padded_h = height + 2 * pad_h;
        let padded_w = width + 2 * pad_w;

        // Calculate output dimensions
        let h_out = (padded_h - kernel_h) / stride_h + 1;
        let w_out = (padded_w - kernel_w) / stride_w + 1;

        // Create padded input (virtually via index computation)
        let x_data_cpu = x_data.to_vec();
        let w_data_cpu = w_data.to_vec();

        // Output tensor
        let output_size = batch * out_ch * h_out * w_out;
        let mut output = vec![0.0f32; output_size];

        // Direct convolution: for each output pixel, compute dot product
        for b in 0..batch {
            for oc in 0..out_ch {
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let mut sum = 0.0f32;

                        // Starting position in input (accounting for padding)
                        let h_start = oh * stride_h;
                        let w_start = ow * stride_w;

                        // Compute dot product over receptive field
                        for ic in 0..in_ch {
                            for kh in 0..kernel_h {
                                for kw in 0..kernel_w {
                                    let h_pos = h_start + kh;
                                    let w_pos = w_start + kw;

                                    // Check if within padded bounds
                                    let in_h = h_pos as isize - pad_h as isize;
                                    let in_w = w_pos as isize - pad_w as isize;

                                    if in_h >= 0
                                        && in_h < height as isize
                                        && in_w >= 0
                                        && in_w < width as isize
                                    {
                                        let in_h = in_h as usize;
                                        let in_w = in_w as usize;

                                        // Input index: (b, ic, in_h, in_w)
                                        let in_idx = b * (in_ch * height * width)
                                            + ic * (height * width)
                                            + in_h * width
                                            + in_w;

                                        // Weight index: (oc, ic, kh, kw)
                                        let w_idx = oc * (in_ch * kernel_h * kernel_w)
                                            + ic * (kernel_h * kernel_w)
                                            + kh * kernel_w
                                            + kw;

                                        if let (Some(&in_val), Some(&w_val)) =
                                            (x_data_cpu.get(in_idx), w_data_cpu.get(w_idx))
                                        {
                                            sum += in_val * w_val;
                                        }
                                    }
                                }
                            }
                        }

                        // Output index: (b, oc, oh, ow)
                        let out_idx =
                            b * (out_ch * h_out * w_out) + oc * (h_out * w_out) + oh * w_out + ow;
                        if let Some(slot) = output.get_mut(out_idx) {
                            *slot = sum;
                        }
                    }
                }
            }
        }

        // Create output tensor
        let result = RawTensor::new(output, &[batch, out_ch, h_out, w_out], x_requires_grad);

        // Attach gradient function if input requires gradients
        if x_requires_grad {
            result.borrow_mut().parents = vec![x.clone(), weight.clone()];
            result.borrow_mut().grad_fn = Some(Box::new(DirectConvGradFn {
                input_shape: x_shape.clone(),
                weight_shape: w_shape.clone(),
                padding: (pad_h, pad_w),
                stride: (stride_h, stride_w),
            }));
        }

        result
    }

    /// GPU Direct convolution helper
    #[allow(clippy::too_many_arguments)]
    fn gpu_direct_conv_internal(
        x: &Tensor,
        weight: &Tensor,
        batch: usize,
        in_channels: usize,
        out_channels: usize,
        height: usize,
        width: usize,
        kernel_h: usize,
        kernel_w: usize,
        pad_h: usize,
        pad_w: usize,
        stride_h: usize,
        stride_w: usize,
    ) -> Tensor {
        let h_out = (height + 2 * pad_h - kernel_h) / stride_h + 1;
        let w_out = (width + 2 * pad_w - kernel_w) / stride_w + 1;

        let x_data = x.borrow().data.clone();
        let w_data = weight.borrow().data.clone();

        #[cfg(feature = "gpu")]
        let result = {
            RawTensor::gpu_direct_conv(
                &x_data,
                &w_data,
                batch,
                in_channels,
                out_channels,
                height,
                width,
                kernel_h,
                kernel_w,
                stride_h,
                stride_w,
                pad_h,
                pad_w,
                h_out,
                w_out,
            )
        };

        #[cfg(not(feature = "gpu"))]
        let result = None;
        if let Some(storage) = result {
            let device = x.borrow().device.clone();
            let requires_grad = x.borrow().requires_grad;
            let raw = RawTensor::new_with_storage(
                storage,
                &[batch, out_channels, h_out, w_out],
                device,
                requires_grad,
            );

            // Attach gradient function if input requires gradients
            if requires_grad {
                let x_shape = x.borrow().shape.clone();
                let w_shape = weight.borrow().shape.clone();

                raw.borrow_mut().parents = vec![x.clone(), weight.clone()];
                raw.borrow_mut().grad_fn = Some(Box::new(DirectConvGradFn {
                    input_shape: x_shape,
                    weight_shape: w_shape,
                    padding: (pad_h, pad_w),
                    stride: (stride_h, stride_w),
                }));
            }

            raw
        } else {
            eprintln!("[WARNING] GPU Direct Convolution failed, falling back to CPU");
            Self::direct_conv(x, weight, pad_h, pad_w, stride_h, stride_w)
        }
    }

    /// Implicit GEMM (iGEMM) convolution - performs GEMM without materializing im2col matrix
    ///
    /// Uses tiled computation for cache efficiency. The algorithm computes output tiles
    /// by loading relevant portions of input and weight tensors, avoiding the memory
    /// overhead of creating the full im2col matrix.
    ///
    /// # Algorithm
    ///
    /// For each output tile `(oc_tile, spatial_tile)`:
    /// 1. For each input channel tile:
    ///    - Load input patch tile: `(spatial_tile, kernel_h * kernel_w)`
    ///    - Load weight tile: `(oc_tile, c_tile * kernel_h * kernel_w)`
    ///    - Compute partial GEMM and accumulate
    /// 2. Store output tile
    ///
    /// This approach:
    /// - Reduces memory bandwidth (no im2col materialization)
    /// - Improves cache locality (tiled computation)
    /// - Enables future optimizations (Winograd, FFT)
    fn igemm_forward(
        x: &Tensor,
        weight: &Tensor,
        pad_h: usize,
        pad_w: usize,
        stride_h: usize,
        stride_w: usize,
        variant: IGEMMVariant,
    ) -> Tensor {
        let (x_data, x_shape, x_device, x_requires_grad) = {
            let x_borrow = x.borrow();
            (
                x_borrow.data.clone(),
                x_borrow.shape.clone(),
                x_borrow.device.clone(),
                x_borrow.requires_grad,
            )
        };

        let (w_data, w_shape) = {
            let w_borrow = weight.borrow();
            (w_borrow.data.clone(), w_borrow.shape.clone())
        };

        let (batch, in_ch, height, width) = (
            x_shape.first().copied().unwrap_or(1),
            x_shape.get(1).copied().unwrap_or(1),
            x_shape.get(2).copied().unwrap_or(1),
            x_shape.get(3).copied().unwrap_or(1),
        );

        let (out_ch, _, kernel_h, kernel_w) = (
            w_shape.first().copied().unwrap_or(1),
            w_shape.get(1).copied().unwrap_or(1),
            w_shape.get(2).copied().unwrap_or(1),
            w_shape.get(3).copied().unwrap_or(1),
        );

        // Apply padding
        let padded_h = height + 2 * pad_h;
        let padded_w = width + 2 * pad_w;

        // Calculate output dimensions
        let h_out = (padded_h - kernel_h) / stride_h + 1;
        let w_out = (padded_w - kernel_w) / stride_w + 1;

        // Try GPU path first
        #[cfg(feature = "gpu")]
        let gpu_result = {
            let is_gpu = matches!(x_device, Device::GPU(_));
            if is_gpu {
                RawTensor::gpu_igemm(
                    &x_data, &w_data, batch, in_ch, out_ch, height, width, kernel_h, kernel_w,
                    stride_h, stride_w, pad_h, pad_w, h_out, w_out,
                )
            } else {
                None
            }
        };

        #[cfg(not(feature = "gpu"))]
        let gpu_result = None;

        if let Some(storage) = gpu_result {
            // Create GPU tensor with grad function
            let result = RawTensor::new_with_storage(
                storage,
                &[batch, out_ch, h_out, w_out],
                x_device.clone(),
                x_requires_grad,
            );

            // Attach gradient function if input requires gradients
            if x_requires_grad {
                result.borrow_mut().parents = vec![x.clone(), weight.clone()];
                result.borrow_mut().grad_fn = Some(Box::new(IGEMMGradFn {
                    input_shape: x_shape,
                    weight_shape: w_shape,
                    padding: (pad_h, pad_w),
                    stride: (stride_h, stride_w),
                    _variant: variant,
                }));
            }

            result
        } else {
            // Fall back to CPU path
            let x_data_cpu = x_data.to_vec();
            let w_data_cpu = w_data.to_vec();

            // Output tensor
            let output_size = batch * out_ch * h_out * w_out;
            let mut output = vec![0.0f32; output_size];

            // Select variant implementation
            match variant {
                IGEMMVariant::Tiled => {
                    let config = TileConfig::default();
                    Self::igemm_tiled(
                        &x_data_cpu,
                        &w_data_cpu,
                        &mut output,
                        batch,
                        in_ch,
                        out_ch,
                        height,
                        width,
                        kernel_h,
                        kernel_w,
                        stride_h,
                        stride_w,
                        pad_h,
                        pad_w,
                        h_out,
                        w_out,
                        &config,
                    );
                }
            }

            // Create output tensor
            let result = RawTensor::new(output, &[batch, out_ch, h_out, w_out], x_requires_grad);

            // Attach gradient function if input requires gradients
            if x_requires_grad {
                result.borrow_mut().parents = vec![x.clone(), weight.clone()];
                result.borrow_mut().grad_fn = Some(Box::new(IGEMMGradFn {
                    input_shape: x_shape,
                    weight_shape: w_shape,
                    padding: (pad_h, pad_w),
                    stride: (stride_h, stride_w),
                    _variant: variant,
                }));
            }

            result
        }
    }

    /// Tiled iGEMM implementation
    ///
    /// Computes convolution by tiling across output channels and spatial dimensions,
    /// iterating over input channels to accumulate partial results.
    #[allow(clippy::too_many_arguments)]
    fn igemm_tiled(
        x_data: &[f32],
        w_data: &[f32],
        output: &mut [f32],
        batch: usize,
        in_ch: usize,
        out_ch: usize,
        height: usize,
        width: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
        pad_h: usize,
        pad_w: usize,
        h_out: usize,
        w_out: usize,
        config: &TileConfig,
    ) {
        // let _kernel_size = kernel_h * kernel_w; kept for reference
        let spatial_out = h_out * w_out;

        // Tile over output channels
        for oc_start in (0..out_ch).step_by(config.output_channels) {
            let oc_end = (oc_start + config.output_channels).min(out_ch);

            // Tile over spatial positions (batch * h_out * w_out)
            for s_start in (0..batch * spatial_out).step_by(config.output_spacial_dims) {
                let s_end = (s_start + config.output_spacial_dims).min(batch * spatial_out);

                // Process input channels in tiles
                for ic_start in (0..in_ch).step_by(config.input_channels) {
                    let ic_end = (ic_start + config.input_channels).min(in_ch);

                    // Compute output tile: [oc_tile, spatial_tile] x [ic_tile * kernel_size]^T
                    // For each output position in tile:
                    for s_idx in s_start..s_end {
                        // Decode spatial index
                        let b = s_idx / spatial_out;
                        let remaining = s_idx % spatial_out;
                        let oh = remaining / w_out;
                        let ow = remaining % w_out;

                        // Starting position in input (accounting for padding)
                        let h_start = oh * stride_h;
                        let w_start = ow * stride_w;

                        // For each output channel in tile:
                        for oc in oc_start..oc_end {
                            let mut sum = 0.0;

                            // Accumulate over input channel tile
                            for ic in ic_start..ic_end {
                                // For each kernel position:
                                for kh in 0..kernel_h {
                                    for kw in 0..kernel_w {
                                        let h_pos = h_start + kh;
                                        let w_pos = w_start + kw;

                                        // Check if within padded bounds
                                        let in_h = h_pos as isize - pad_h as isize;
                                        let in_w = w_pos as isize - pad_w as isize;

                                        if in_h >= 0
                                            && in_h < height as isize
                                            && in_w >= 0
                                            && in_w < width as isize
                                        {
                                            let in_h = in_h as usize;
                                            let in_w = in_w as usize;

                                            // Input index: (b, ic, in_h, in_w)
                                            let in_idx = b * (in_ch * height * width)
                                                + ic * (height * width)
                                                + in_h * width
                                                + in_w;

                                            // Weight index: (oc, ic, kh, kw)
                                            let w_idx = oc * (in_ch * kernel_h * kernel_w)
                                                + ic * (kernel_h * kernel_w)
                                                + kh * kernel_w
                                                + kw;

                                            if let (Some(&in_val), Some(&w_val)) =
                                                (x_data.get(in_idx), w_data.get(w_idx))
                                            {
                                                sum += in_val * w_val;
                                            }
                                        }
                                    }
                                }
                            }

                            // Output index: (b, oc, oh, ow)
                            let out_idx = b * (out_ch * spatial_out) + oc * spatial_out + s_idx;
                            if let Some(slot) = output.get_mut(out_idx) {
                                *slot += sum;
                            }
                        }
                    }
                }
            }
        }
    }

    /// # Panics
    /// Input needs to be 4D
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let (batch, in_channels, height, width) = {
            let x_borrow = x.borrow();
            assert_eq!(x_borrow.shape.len(), 4, "Input must be 4D: (B, C, H, W)");
            (
                x_borrow.shape.first().copied().unwrap_or(1),
                x_borrow.shape.get(1).copied().unwrap_or(1),
                x_borrow.shape.get(2).copied().unwrap_or(1),
                x_borrow.shape.get(3).copied().unwrap_or(1),
            )
        };

        let (out_channels, kernel_h, kernel_w) = {
            let w_borrow = self.weight.borrow();
            assert_eq!(
                w_borrow.shape.get(1).copied().unwrap_or(1),
                in_channels,
                "Channel mismatch"
            );
            (
                w_borrow.shape.first().copied().unwrap_or(1),
                w_borrow.shape.get(2).copied().unwrap_or(1),
                w_borrow.shape.get(3).copied().unwrap_or(1),
            )
        };

        let (pad_h, pad_w) = self.padding;
        let (stride_h, stride_w) = self.stride;

        // Check if gradients are needed
        let _requires_grad = x.borrow().requires_grad;

        // Algorithm selection
        // Note: All algorithms now support gradients!
        let algo = self.algo.get();
        let algo = match algo {
            ConvAlgo::Auto => ConvAlgo::auto_select(
                batch,
                in_channels,
                height,
                width,
                kernel_h,
                x.borrow().device.is_gpu(),
            ),
            other @ (ConvAlgo::Direct | ConvAlgo::Im2col | ConvAlgo::IGEMM) => other,
        };

        // Compute output using selected algorithm
        let out = match algo {
            ConvAlgo::Direct => {
                // Check if we're on GPU
                if x.borrow().device.is_gpu() {
                    // GPU Direct Convolution
                    Self::gpu_direct_conv_internal(
                        x,
                        &self.weight,
                        batch,
                        in_channels,
                        out_channels,
                        height,
                        width,
                        kernel_h,
                        kernel_w,
                        pad_h,
                        pad_w,
                        stride_h,
                        stride_w,
                    )
                } else {
                    // CPU Direct Convolution (existing)
                    Self::direct_conv(x, &self.weight, pad_h, pad_w, stride_h, stride_w)
                }
            }
            ConvAlgo::IGEMM => {
                // Implicit GEMM convolution (CPU only for now)
                Self::igemm_forward(
                    x,
                    &self.weight,
                    pad_h,
                    pad_w,
                    stride_h,
                    stride_w,
                    IGEMMVariant::default(),
                )
            }
            ConvAlgo::Im2col | ConvAlgo::Auto => {
                // Auto already converted, but kept for exhaustiveness
                // 1. Pad input
                let x_padded = if pad_h > 0 || pad_w > 0 {
                    x.pad(&[(0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)])
                } else {
                    x.clone()
                };

                let (padded_h, padded_w) = {
                    let p = x_padded.borrow();
                    (
                        p.shape.get(2).copied().unwrap_or(1),
                        p.shape.get(3).copied().unwrap_or(1),
                    )
                };

                // Calculate output dimensions
                let h_out = (padded_h - kernel_h) / stride_h + 1;
                let w_out = (padded_w - kernel_w) / stride_w + 1;

                // 2. Apply im2col: (B, C, H_pad, W_pad) → (B*H_out*W_out, C*K*K)
                let col = Self::im2col(&x_padded, (kernel_h, kernel_w), (stride_h, stride_w));

                // 3. Reshape weights: (O, C, K, K) → (C*K*K, O)
                let weight_2d = self
                    .weight
                    .reshape(&[out_channels, in_channels * kernel_h * kernel_w]);
                let weight_t = weight_2d.permute(&[1, 0]); // Transpose to (C*K*K, O)

                // 4. Matmul: (B*H_out*W_out, C*K*K) @ (C*K*K, O) → (B*H_out*W_out, O)
                let out_2d = col.matmul(&weight_t);

                // 5. Reshape: (B*H_out*W_out, O) → (B, H_out, W_out, O) → (B, O, H_out, W_out)
                let out_4d = out_2d.reshape(&[batch, h_out, w_out, out_channels]);
                out_4d.permute(&[0, 3, 1, 2]) // (B, O, H_out, W_out)
            }
        };

        // Add bias if present (common to both algorithms)
        let h_out = out.borrow().shape.get(2).copied().unwrap_or(1);
        let w_out = out.borrow().shape.get(3).copied().unwrap_or(1);

        if let Some(ref b) = self.bias {
            // Reshape bias from (out_channels,) to (1, out_channels, 1, 1)
            let bias_reshaped = b.reshape(&[1, out_channels, 1, 1]);

            // Workaround: GPU broadcasting not yet implemented for binary ops
            // Manually expand bias to match output shape
            let bias_expanded = bias_reshaped.expand(&[batch, out_channels, h_out, w_out]);

            out.add(&bias_expanded)
        } else {
            out
        }
    }
}

impl Module for Conv2d {
    fn forward(&self, x: &Tensor) -> Tensor {
        self.forward(x)
    }
    fn parameters(&self) -> Vec<Tensor> {
        let mut p = vec![self.weight.clone()];
        if let Some(ref b) = self.bias {
            p.push(b.clone());
        }
        p
    }

    fn state_dict(&self) -> StateDict {
        let mut state = StateDict::new();
        state.insert("weight".to_string(), TensorData::from_tensor(&self.weight));
        if let Some(ref b) = self.bias {
            state.insert("bias".to_string(), TensorData::from_tensor(b));
        }
        state
    }

    fn load_state_dict(&mut self, state: &StateDict) {
        if let Some(w) = state.get("weight") {
            let mut t = self.weight.borrow_mut();
            t.data = Storage::cpu(w.data.clone());
            t.shape.clone_from(&w.shape);
        }
        // Load bias if state has it and layer has bias
        if let (Some(b), Some(bias_tensor)) = (state.get("bias"), self.bias.as_ref()) {
            let mut t = bias_tensor.borrow_mut();
            t.data = Storage::cpu(b.data.clone());
            t.shape.clone_from(&b.shape);
        }
    }
}

#[cfg(test)]
mod conv2d_tests {
    use super::*;

    #[test]
    fn test_conv2d_forward_shape() {
        // Input: (1, 3, 32, 32), Conv: 16 filters, 3x3, stride=1, pad=1
        // Output should be (1, 16, 32, 32)
        let conv = Conv2d::new(3, 16, 3, 1, 1, true);
        let x = RawTensor::randn(&[1, 3, 32, 32]);
        let y = conv.forward(&x);

        assert_eq!(y.borrow().shape, vec![1, 16, 32, 32]);
    }

    #[test]
    fn test_conv2d_forward_shape_no_padding() {
        // Input: (2, 3, 8, 8), Conv: 8 filters, 3x3, stride=1, pad=0
        // Output: (2, 8, 6, 6) since (8 - 3) / 1 + 1 = 6
        let conv = Conv2d::new(3, 8, 3, 1, 0, false);
        let x = RawTensor::randn(&[2, 3, 8, 8]);
        let y = conv.forward(&x);

        assert_eq!(y.borrow().shape, vec![2, 8, 6, 6]);
    }

    #[test]
    fn test_conv2d_forward_shape_stride2() {
        // Input: (1, 1, 8, 8), Conv: 4 filters, 3x3, stride=2, pad=1
        // Output: (1, 4, 4, 4) since (8 + 2 - 3) / 2 + 1 = 4
        let conv = Conv2d::new(1, 4, 3, 2, 1, true);
        let x = RawTensor::randn(&[1, 1, 8, 8]);
        let y = conv.forward(&x);

        assert_eq!(y.borrow().shape, vec![1, 4, 4, 4]);
    }

    #[test]
    //CARE: can sometimes fail as well, need to fix
    fn test_conv2d_gradient() {
        // Use smaller input for more stable numerical gradient checking
        let conv = Conv2d::new(2, 4, 3, 1, 1, true);
        let x = RawTensor::randn(&[1, 2, 6, 6]);
        x.borrow_mut().requires_grad = true;

        // Convolution is affine in its inputs, but each loss evaluation sums tens of thousands
        // of f32 multiply-adds (im2col + GEMM). With ε=1e-2 the central-difference estimator
        // was dominated by round-off (~1e-5) and produced ~3% relative error. Increasing ε
        // lowers that amplification without changing the true derivative.
        let (max_err, mean_err, passed) = RawTensor::check_gradients(
            &x,
            |t| conv.forward(t).sum(),
            5e-2, // epsilon, less cancellation noise for heavy Conv2d graphs
            2e-2, // tolerance (relaxed for conv's numerical complexity)
        );

        assert!(
            passed,
            "Conv2d gradient check failed: max_error={max_err:.6e}, mean_error={mean_err:.6e}"
        );
    }

    #[test]
    fn test_conv2d_backward_flow() {
        // Test that gradients flow through the entire network
        let conv = Conv2d::new(2, 4, 3, 1, 0, true);
        let x = RawTensor::randn(&[1, 2, 5, 5]);
        x.borrow_mut().requires_grad = true;

        let y = conv.forward(&x);
        let loss = y.sum();
        loss.backward();

        // Check that input has gradients
        assert!(x.grad().is_some(), "Input should have gradients");

        // Check that weights have gradients
        assert!(
            conv.weight.grad().is_some(),
            "Weights should have gradients"
        );

        // Check that bias has gradients
        if let Some(ref b) = conv.bias {
            assert!(b.grad().is_some(), "Bias should have gradients");
        }
    }

    #[test]
    fn test_conv2d_no_bias() {
        // Test convolution without bias
        let conv = Conv2d::new(1, 2, 3, 1, 0, false);
        let x = RawTensor::randn(&[1, 1, 5, 5]);
        let y = conv.forward(&x);

        assert_eq!(y.borrow().shape, vec![1, 2, 3, 3]);
        assert!(conv.bias.is_none(), "Bias should be None");
    }

    // ===== Comprehensive Parameter Configuration Tests =====

    #[test]
    fn test_conv2d_1x1_kernel() {
        // 1x1 convolution is equivalent to per-pixel linear transformation
        // Input: (1, 4, 8, 8), Conv: 8 filters, 1x1, stride=1, pad=0
        // Output: (1, 8, 8, 8)
        let conv = Conv2d::new(4, 8, 1, 1, 0, true);
        let x = RawTensor::randn(&[1, 4, 8, 8]);
        let y = conv.forward(&x);

        assert_eq!(y.borrow().shape, vec![1, 8, 8, 8]);
    }

    #[test]
    fn test_conv2d_5x5_kernel() {
        // Larger kernel size
        // Input: (1, 3, 16, 16), Conv: 8 filters, 5x5, stride=1, pad=2
        // Output: (1, 8, 16, 16) since (16 + 4 - 5) / 1 + 1 = 16
        let conv = Conv2d::new(3, 8, 5, 1, 2, true);
        let x = RawTensor::randn(&[1, 3, 16, 16]);
        let y = conv.forward(&x);

        assert_eq!(y.borrow().shape, vec![1, 8, 16, 16]);
    }

    #[test]
    fn test_conv2d_7x7_kernel() {
        // Even larger kernel
        // Input: (1, 3, 32, 32), Conv: 16 filters, 7x7, stride=1, pad=3
        // Output: (1, 16, 32, 32)
        let conv = Conv2d::new(3, 16, 7, 1, 3, true);
        let x = RawTensor::randn(&[1, 3, 32, 32]);
        let y = conv.forward(&x);

        assert_eq!(y.borrow().shape, vec![1, 16, 32, 32]);
    }

    #[test]
    fn test_conv2d_stride3() {
        // Large stride
        // Input: (1, 3, 16, 16), Conv: 8 filters, 3x3, stride=3, pad=1
        // Output: (1, 8, 6, 6) since (16 + 2 - 3) / 3 + 1 = 6
        let conv = Conv2d::new(3, 8, 3, 3, 1, true);
        let x = RawTensor::randn(&[1, 3, 16, 16]);
        let y = conv.forward(&x);

        assert_eq!(y.borrow().shape, vec![1, 8, 6, 6]);
    }

    #[test]
    fn test_conv2d_stride4() {
        // Very large stride
        // Input: (1, 3, 32, 32), Conv: 8 filters, 3x3, stride=4, pad=1
        // Output: (1, 8, 8, 8) since (32 + 2 - 3) / 4 + 1 = 8
        let conv = Conv2d::new(3, 8, 3, 4, 1, true);
        let x = RawTensor::randn(&[1, 3, 32, 32]);
        let y = conv.forward(&x);

        assert_eq!(y.borrow().shape, vec![1, 8, 8, 8]);
    }

    #[test]
    fn test_conv2d_padding2() {
        // Larger padding
        // Input: (1, 3, 8, 8), Conv: 8 filters, 3x3, stride=1, pad=2
        // Output: (1, 8, 10, 10) since (8 + 4 - 3) / 1 + 1 = 10
        let conv = Conv2d::new(3, 8, 3, 1, 2, true);
        let x = RawTensor::randn(&[1, 3, 8, 8]);
        let y = conv.forward(&x);

        assert_eq!(y.borrow().shape, vec![1, 8, 10, 10]);
    }

    #[test]
    fn test_conv2d_padding3() {
        // Even larger padding
        // Input: (1, 3, 8, 8), Conv: 8 filters, 3x3, stride=1, pad=3
        // Output: (1, 8, 12, 12) since (8 + 6 - 3) / 1 + 1 = 12
        let conv = Conv2d::new(3, 8, 3, 1, 3, true);
        let x = RawTensor::randn(&[1, 3, 8, 8]);
        let y = conv.forward(&x);

        assert_eq!(y.borrow().shape, vec![1, 8, 12, 12]);
    }

    #[test]
    fn test_conv2d_large_stride_small_output() {
        // Large stride causing very small output
        // Input: (1, 3, 32, 32), Conv: 8 filters, 5x5, stride=4, pad=2
        // Output: (1, 8, 8, 8) since (32 + 4 - 5) / 4 + 1 = 8
        let conv = Conv2d::new(3, 8, 5, 4, 2, true);
        let x = RawTensor::randn(&[1, 3, 32, 32]);
        let y = conv.forward(&x);

        assert_eq!(y.borrow().shape, vec![1, 8, 8, 8]);
    }

    #[test]
    fn test_conv2d_different_channels() {
        // Test various input/output channel combinations
        let test_cases = vec![
            (1, 1, 8, 8),     // 1->1
            (1, 16, 8, 8),    // 1->16
            (3, 64, 16, 16),  // 3->64
            (64, 128, 8, 8),  // 64->128
            (128, 256, 4, 4), // 128->256
        ];

        for (in_ch, out_ch, h, w) in test_cases {
            let conv = Conv2d::new(in_ch, out_ch, 3, 1, 1, true);
            let x = RawTensor::randn(&[1, in_ch, h, w]);
            let y = conv.forward(&x);

            assert_eq!(
                y.borrow().shape,
                vec![1, out_ch, h, w],
                "Failed for in_ch={in_ch}, out_ch={out_ch}"
            );
        }
    }

    #[test]
    fn test_conv2d_multiple_batch_sizes() {
        // Test different batch sizes
        let batch_sizes = vec![1, 2, 4, 8, 16];

        for batch in batch_sizes {
            let conv = Conv2d::new(3, 16, 3, 1, 1, true);
            let x = RawTensor::randn(&[batch, 3, 32, 32]);
            let y = conv.forward(&x);

            assert_eq!(
                y.borrow().shape,
                vec![batch, 16, 32, 32],
                "Failed for batch_size={batch}"
            );
        }
    }

    #[test]
    fn test_conv2d_small_input() {
        // Test with minimal valid input sizes
        // Input: (1, 2, 4, 4), Conv: 4 filters, 3x3, stride=1, pad=1
        // Output: (1, 4, 4, 4)
        let conv = Conv2d::new(2, 4, 3, 1, 1, true);
        let x = RawTensor::randn(&[1, 2, 4, 4]);
        let y = conv.forward(&x);

        assert_eq!(y.borrow().shape, vec![1, 4, 4, 4]);
    }

    #[test]
    fn test_conv2d_asymmetric_dimensions() {
        // Test with non-square spatial dimensions
        // Input: (1, 3, 16, 32), Conv: 8 filters, 3x3, stride=2, pad=1
        // Output: (1, 8, 8, 16)
        let conv = Conv2d::new(3, 8, 3, 2, 1, true);
        let x = RawTensor::randn(&[1, 3, 16, 32]);
        let y = conv.forward(&x);

        assert_eq!(y.borrow().shape, vec![1, 8, 8, 16]);
    }

    #[test]
    fn test_conv2d_5x5_stride2_padding2() {
        // Classic architecture combination (e.g., early VGG layers)
        // Input: (1, 3, 32, 32), Conv: 64 filters, 5x5, stride=2, pad=2
        // Output: (1, 64, 16, 16)
        let conv = Conv2d::new(3, 64, 5, 2, 2, true);
        let x = RawTensor::randn(&[1, 3, 32, 32]);
        let y = conv.forward(&x);

        assert_eq!(y.borrow().shape, vec![1, 64, 16, 16]);
    }

    #[test]
    fn test_conv2d_output_shape_calculation() {
        // Verify output shape formula: (H + 2*pad - kernel) / stride + 1
        let configs = vec![
            // (in_h, in_w, kernel, stride, pad, expected_h, expected_w)
            (32, 32, 3, 1, 1, 32, 32),
            (32, 32, 3, 2, 1, 16, 16),
            (28, 28, 5, 1, 2, 28, 28),
            (28, 28, 5, 2, 2, 14, 14),
            (14, 14, 3, 1, 1, 14, 14),
            (8, 8, 3, 2, 1, 4, 4),
            (16, 16, 7, 2, 3, 8, 8), // (16 + 6 - 7) / 2 + 1 = 15 / 2 + 1 = 8
        ];

        for (in_h, in_w, kernel, stride, pad, exp_h, exp_w) in configs {
            let conv = Conv2d::new(3, 8, kernel, stride, pad, true);
            let x = RawTensor::randn(&[1, 3, in_h, in_w]);
            let y = conv.forward(&x);

            assert_eq!(
                y.borrow().shape,
                vec![1, 8, exp_h, exp_w],
                "Failed for input=({in_h},{in_w}), kernel={kernel}, stride={stride}, pad={pad}"
            );
        }
    }

    // ===== Gradient Tests for Various Configurations =====

    #[test]
    fn test_conv2d_gradient_1x1() {
        // Gradient check for 1x1 convolution
        let conv = Conv2d::new(2, 4, 1, 1, 0, true);
        let x = RawTensor::randn(&[1, 2, 4, 4]);
        x.borrow_mut().requires_grad = true;

        let (max_err, mean_err, passed) =
            RawTensor::check_gradients(&x, |t| conv.forward(t).sum(), 5e-2, 2e-2);

        assert!(
            passed,
            "Conv2d 1x1 gradient check failed: max_error={max_err:.6e}, mean_error={mean_err:.6e}"
        );
    }

    #[test]
    fn test_conv2d_gradient_5x5() {
        // Gradient check for 5x5 convolution
        let conv = Conv2d::new(2, 4, 5, 1, 2, true);
        let x = RawTensor::randn(&[1, 2, 8, 8]);
        x.borrow_mut().requires_grad = true;

        let (max_err, mean_err, passed) =
            RawTensor::check_gradients(&x, |t| conv.forward(t).sum(), 5e-2, 2e-2);

        assert!(
            passed,
            "Conv2d 5x5 gradient check failed: max_error={max_err:.6e}, mean_error={mean_err:.6e}"
        );
    }

    #[test]
    fn test_conv2d_gradient_stride2() {
        // Gradient check with stride 2
        let conv = Conv2d::new(2, 4, 3, 2, 1, true);
        let x = RawTensor::randn(&[1, 2, 8, 8]);
        x.borrow_mut().requires_grad = true;

        let (max_err, mean_err, passed) =
            RawTensor::check_gradients(&x, |t| conv.forward(t).sum(), 5e-2, 2e-2);

        assert!(
            passed,
            "Conv2d stride2 gradient check failed: max_error={max_err:.6e}, mean_error={mean_err:.6e}"
        );
    }

    #[test]
    fn test_conv2d_gradient_no_bias() {
        // Gradient check without bias
        let conv = Conv2d::new(2, 4, 3, 1, 1, false);
        let x = RawTensor::randn(&[1, 2, 6, 6]);
        x.borrow_mut().requires_grad = true;

        let (max_err, mean_err, passed) =
            RawTensor::check_gradients(&x, |t| conv.forward(t).sum(), 5e-2, 2e-2);

        assert!(
            passed,
            "Conv2d no-bias gradient check failed: max_error={max_err:.6e}, mean_error={mean_err:.6e}"
        );
    }

    #[test]
    fn test_conv2d_gradient_multiple_channels() {
        // Gradient check with more channels
        let conv = Conv2d::new(4, 8, 3, 1, 1, true);
        let x = RawTensor::randn(&[1, 4, 8, 8]);
        x.borrow_mut().requires_grad = true;

        let (max_err, mean_err, passed) =
            RawTensor::check_gradients(&x, |t| conv.forward(t).sum(), 5e-2, 2e-2);

        assert!(
            passed,
            "Conv2d multi-channel gradient check failed: max_error={max_err:.6e}, mean_error={mean_err:.6e}"
        );
    }

    #[test]
    fn test_conv2d_parameter_gradients_shape() {
        // Verify that parameter gradients have correct shapes
        let conv = Conv2d::new(3, 16, 3, 1, 1, true);
        let x = RawTensor::randn(&[2, 3, 32, 32]);
        x.borrow_mut().requires_grad = true;

        let y = conv.forward(&x);
        let loss = y.sum();
        loss.backward();

        // Check weight gradient shape
        if let Some(w_grad) = conv.weight.grad() {
            assert_eq!(w_grad.len(), 3 * 16 * 3 * 3, "Weight grad has wrong size");
        } else {
            panic!("Weight gradient is None");
        }

        // Check bias gradient shape
        if let Some(ref b) = conv.bias {
            if let Some(b_grad) = b.grad() {
                assert_eq!(b_grad.len(), 16, "Bias grad has wrong size");
            } else {
                panic!("Bias gradient is None");
            }
        }

        // Check input gradient shape
        if let Some(x_grad) = x.grad() {
            assert_eq!(x_grad.len(), 2 * 3 * 32 * 32, "Input grad has wrong size");
        } else {
            panic!("Input gradient is None");
        }
    }

    #[test]
    fn test_conv2d_direct_conv_inference() {
        // Test that direct convolution produces correct results for inference
        let conv = Conv2d::new(2, 4, 3, 1, 1, true);
        // Force direct algorithm (no gradients, so direct will be used)
        conv.algo.set(ConvAlgo::Direct);

        let x = RawTensor::randn(&[1, 2, 8, 8]);
        // Don't set requires_grad - direct conv will be used

        let y_direct = conv.forward(&x);

        // Compare with im2col
        conv.algo.set(ConvAlgo::Im2col);
        let y_im2col = conv.forward(&x);

        // Results should be identical
        assert_eq!(y_direct.borrow().shape, y_im2col.borrow().shape);

        let y_direct_data = y_direct.borrow().data.to_vec();
        let y_im2col_data = y_im2col.borrow().data.to_vec();

        assert_eq!(y_direct_data.len(), y_im2col_data.len());

        for (i, (direct_val, im2col_val)) in
            y_direct_data.iter().zip(y_im2col_data.iter()).enumerate()
        {
            let abs_diff = (direct_val - im2col_val).abs();
            assert!(
                abs_diff < 1e-5,
                "Direct vs im2col mismatch at index {i}: direct={direct_val}, im2col={im2col_val}, diff={abs_diff}"
            );
        }
    }

    #[test]
    fn test_conv2d_direct_conv_with_padding() {
        // Test direct convolution with padding
        let conv = Conv2d::new(2, 4, 3, 1, 2, true);
        conv.algo.set(ConvAlgo::Direct);

        let x = RawTensor::new(
            (0..32).map(|i| i as f32).collect::<Vec<_>>(),
            &[1, 2, 4, 4],
            false,
        );

        let y_direct = conv.forward(&x);

        // Compare with im2col
        conv.algo.set(ConvAlgo::Im2col);
        let y_im2col = conv.forward(&x);

        let y_direct_data = y_direct.borrow().data.to_vec();
        let y_im2col_data = y_im2col.borrow().data.to_vec();

        for (i, (direct_val, im2col_val)) in
            y_direct_data.iter().zip(y_im2col_data.iter()).enumerate()
        {
            let abs_diff = (direct_val - im2col_val).abs();
            assert!(
                abs_diff < 1e-5,
                "Padding test mismatch at index {i}: direct={direct_val}, im2col={im2col_val}, diff={abs_diff}"
            );
        }
    }

    #[test]
    fn test_conv2d_auto_selects_direct_for_small_input() {
        // Test that auto-selection chooses direct for small inputs without gradients
        let conv = Conv2d::new(2, 4, 3, 1, 0, false);
        conv.algo.set(ConvAlgo::Auto);

        // Small input, no gradients - should select Direct
        let x = RawTensor::randn(&[1, 2, 4, 4]);
        let y = conv.forward(&x);

        // Verify output shape is correct
        assert_eq!(y.borrow().shape, vec![1, 4, 2, 2]);
    }

    // ===== Direct Convolution Gradient Tests =====

    #[test]
    fn test_direct_conv_gradient_no_padding() {
        // Test direct conv gradient with no padding
        let conv = Conv2d::new(2, 4, 3, 1, 0, true);
        conv.set_algo(ConvAlgo::Direct);

        let x = RawTensor::randn(&[1, 2, 8, 8]);
        x.borrow_mut().requires_grad = true;

        let (max_err, mean_err, passed) =
            RawTensor::check_gradients(&x, |t| conv.forward(t).sum(), 5e-2, 2e-2);

        assert!(
            passed,
            "Direct conv gradient check failed: max_error={max_err:.6e}, mean_error={mean_err:.6e}"
        );
    }

    #[test]
    fn test_direct_conv_gradient_with_padding() {
        // Test direct conv gradient with padding
        let conv = Conv2d::new(2, 4, 3, 1, 1, true);
        conv.set_algo(ConvAlgo::Direct);

        let x = RawTensor::randn(&[1, 2, 6, 6]);
        x.borrow_mut().requires_grad = true;

        let (max_err, mean_err, passed) =
            RawTensor::check_gradients(&x, |t| conv.forward(t).sum(), 5e-2, 2e-2);

        assert!(
            passed,
            "Direct conv with padding gradient check failed: max_error={max_err:.6e}, mean_error={mean_err:.6e}"
        );
    }

    #[test]
    fn test_direct_conv_gradient_stride2() {
        // Test direct conv gradient with stride 2
        let conv = Conv2d::new(2, 4, 3, 2, 1, true);
        conv.set_algo(ConvAlgo::Direct);

        let x = RawTensor::randn(&[1, 2, 8, 8]);
        x.borrow_mut().requires_grad = true;

        let (max_err, mean_err, passed) =
            RawTensor::check_gradients(&x, |t| conv.forward(t).sum(), 5e-2, 2e-2);

        assert!(
            passed,
            "Direct conv stride2 gradient check failed: max_error={max_err:.6e}, mean_error={mean_err:.6e}"
        );
    }

    #[test]
    fn test_direct_conv_gradient_1x1_kernel() {
        // Test direct conv gradient with 1x1 kernel
        let conv = Conv2d::new(2, 4, 1, 1, 0, true);
        conv.set_algo(ConvAlgo::Direct);

        let x = RawTensor::randn(&[1, 2, 4, 4]);
        x.borrow_mut().requires_grad = true;

        let (max_err, mean_err, passed) =
            RawTensor::check_gradients(&x, |t| conv.forward(t).sum(), 5e-2, 2e-2);

        assert!(
            passed,
            "Direct conv 1x1 gradient check failed: max_error={max_err:.6e}, mean_error={mean_err:.6e}"
        );
    }

    #[test]
    fn test_direct_conv_gradient_no_bias() {
        // Test direct conv gradient without bias
        let conv = Conv2d::new(2, 4, 3, 1, 1, false);
        conv.set_algo(ConvAlgo::Direct);

        let x = RawTensor::randn(&[1, 2, 6, 6]);
        x.borrow_mut().requires_grad = true;

        let (max_err, mean_err, passed) =
            RawTensor::check_gradients(&x, |t| conv.forward(t).sum(), 5e-2, 2e-2);

        assert!(
            passed,
            "Direct conv no-bias gradient check failed: max_error={max_err:.6e}, mean_error={mean_err:.6e}"
        );
    }

    #[test]
    fn test_direct_conv_backward_flow() {
        // Test that gradients flow through direct conv
        let conv = Conv2d::new(2, 4, 3, 1, 1, true);
        conv.set_algo(ConvAlgo::Direct);

        let x = RawTensor::randn(&[1, 2, 6, 6]);
        x.borrow_mut().requires_grad = true;

        let y = conv.forward(&x);
        let loss = y.sum();
        loss.backward();

        // Check that input has gradients
        assert!(x.grad().is_some(), "Input should have gradients");

        // Check that weights have gradients
        assert!(
            conv.weight.grad().is_some(),
            "Weights should have gradients"
        );

        // Check that bias has gradients
        if let Some(ref b) = conv.bias {
            assert!(b.grad().is_some(), "Bias should have gradients");
        }
    }

    #[test]
    fn test_direct_conv_gradient_match_im2col() {
        // Test that direct conv gradients match im2col gradients
        let conv_direct = Conv2d::new(2, 4, 3, 1, 1, true);
        conv_direct.set_algo(ConvAlgo::Direct);

        let conv_im2col = Conv2d::new(2, 4, 3, 1, 1, true);
        conv_im2col.set_algo(ConvAlgo::Im2col);

        // Copy weights from direct to im2col
        let direct_weight_data = conv_direct.weight.borrow().data.to_vec();
        conv_im2col.weight.borrow_mut().data =
            crate::storage::Storage::cpu(direct_weight_data.clone());

        if let (Some(direct_bias), Some(im2col_bias)) = (&conv_direct.bias, &conv_im2col.bias) {
            let direct_bias_data = direct_bias.borrow().data.to_vec();
            im2col_bias.borrow_mut().data = crate::storage::Storage::cpu(direct_bias_data.clone());
        }

        let x = RawTensor::randn(&[1, 2, 6, 6]);
        x.borrow_mut().requires_grad = true;

        // Compute gradients with direct conv
        let y_direct = conv_direct.forward(&x);
        let loss_direct = y_direct.sum();
        loss_direct.backward();

        let x_grad_direct = x.grad().clone();
        let w_grad_direct = conv_direct.weight.grad().clone();

        // Clear gradients
        x.borrow_mut().grad = None;
        conv_direct.weight.borrow_mut().grad = None;
        if let Some(ref b) = conv_direct.bias {
            b.borrow_mut().grad = None;
        }

        // Compute gradients with im2col
        let y_im2col = conv_im2col.forward(&x);
        let loss_im2col = y_im2col.sum();
        loss_im2col.backward();

        let x_grad_im2col = x.grad().clone();
        let w_grad_im2col = conv_im2col.weight.grad().clone();

        // Compare input gradients
        assert!(x_grad_direct.is_some() && x_grad_im2col.is_some());
        let x_grad_direct_data = x_grad_direct.unwrap();
        let x_grad_im2col_data = x_grad_im2col.unwrap();

        assert_eq!(x_grad_direct_data.len(), x_grad_im2col_data.len());

        for (i, (direct_val, im2col_val)) in x_grad_direct_data
            .iter()
            .zip(x_grad_im2col_data.iter())
            .enumerate()
        {
            let abs_diff = (direct_val - im2col_val).abs();
            assert!(
                abs_diff < 1e-4,
                "Input gradient mismatch at index {i}: direct={direct_val}, im2col={im2col_val}, diff={abs_diff}"
            );
        }

        // Compare weight gradients
        assert!(w_grad_direct.is_some() && w_grad_im2col.is_some());
        let w_grad_direct_data = w_grad_direct.unwrap();
        let w_grad_im2col_data = w_grad_im2col.unwrap();

        assert_eq!(w_grad_direct_data.len(), w_grad_im2col_data.len());

        for (i, (direct_val, im2col_val)) in w_grad_direct_data
            .iter()
            .zip(w_grad_im2col_data.iter())
            .enumerate()
        {
            let abs_diff = (direct_val - im2col_val).abs();
            assert!(
                abs_diff < 1e-4,
                "Weight gradient mismatch at index {i}: direct={direct_val}, im2col={im2col_val}, diff={abs_diff}"
            );
        }
    }

    #[test]
    fn test_direct_conv_gradient_multiple_configs() {
        // Test direct conv gradients with various configurations
        let configs = vec![
            (2, 4, 3, 1, 0, 6, 6), // No padding
            (2, 4, 3, 1, 1, 6, 6), // With padding
            (2, 4, 3, 2, 1, 8, 8), // Stride 2
            (2, 4, 1, 1, 0, 4, 4), // 1x1 kernel
            (2, 4, 5, 1, 2, 8, 8), // 5x5 kernel
        ];

        for (in_ch, out_ch, kernel, stride, padding, h, w) in configs {
            let conv = Conv2d::new(in_ch, out_ch, kernel, stride, padding, true);
            conv.set_algo(ConvAlgo::Direct);

            let x = RawTensor::randn(&[1, in_ch, h, w]);
            x.borrow_mut().requires_grad = true;

            let (max_err, mean_err, passed) =
                RawTensor::check_gradients(&x, |t| conv.forward(t).sum(), 5e-2, 2e-2);

            assert!(
                passed,
                "Direct conv gradient check failed for config ({in_ch},{out_ch},{kernel},{stride},{padding}): max_error={max_err:.6e}, mean_error={mean_err:.6e}"
            );
        }
    }

    #[test]
    fn test_conv2d_auto_selects_direct_with_gradients() {
        // Test that auto-selection can now choose Direct even with gradients
        let conv = Conv2d::new(2, 4, 3, 1, 0, true);
        conv.set_algo(ConvAlgo::Auto);

        let x = RawTensor::randn(&[1, 2, 4, 4]);
        x.borrow_mut().requires_grad = true;

        // Should now use Direct (small input, gradients OK)
        let y = conv.forward(&x);
        let loss = y.sum();
        loss.backward();

        // Verify gradients exist
        assert!(x.grad().is_some(), "Input should have gradients");
        assert!(
            conv.weight.grad().is_some(),
            "Weights should have gradients"
        );
    }

    #[test]
    fn test_conv2d_igemm_forward_shape() {
        // Test iGEMM output shape
        let conv = Conv2d::new(3, 16, 3, 1, 1, true);
        conv.set_algo(ConvAlgo::IGEMM);

        let x = RawTensor::randn(&[2, 3, 32, 32]);
        let y = conv.forward(&x);

        assert_eq!(y.borrow().shape, vec![2, 16, 32, 32]);
    }

    #[test]
    fn test_conv2d_igemm_forward_correctness() {
        // Test iGEMM correctness by comparing with Direct
        let x = RawTensor::randn(&[1, 2, 8, 8]);

        let conv_direct = Conv2d::new(2, 4, 3, 1, 1, true);
        conv_direct.set_algo(ConvAlgo::Direct);

        let conv_igemm = Conv2d::new(2, 4, 3, 1, 1, true);
        conv_igemm.set_algo(ConvAlgo::IGEMM);

        // Copy weights
        let weight_data = conv_direct.weight.borrow().data.to_vec();
        conv_igemm.weight.borrow_mut().data = crate::storage::Storage::cpu(weight_data.clone());

        let bias_data = conv_direct.bias.as_ref().unwrap().borrow().data.to_vec();
        conv_igemm.bias.as_ref().unwrap().borrow_mut().data =
            crate::storage::Storage::cpu(bias_data.clone());

        let y_direct = conv_direct.forward(&x);
        let y_igemm = conv_igemm.forward(&x);

        // Compare results
        let y_direct_data = y_direct.borrow().data.to_vec();
        let y_igemm_data = y_igemm.borrow().data.to_vec();

        assert_eq!(y_direct_data.len(), y_igemm_data.len());
        for (i, (direct_val, igemm_val)) in
            y_direct_data.iter().zip(y_igemm_data.iter()).enumerate()
        {
            let abs_diff = (direct_val - igemm_val).abs();
            assert!(
                abs_diff < 1e-5,
                "Mismatch at index {i}: Direct={direct_val}, IGEMM={igemm_val}, diff={abs_diff}"
            );
        }
    }

    #[test]
    fn test_conv2d_igemm_gradient() {
        // Test iGEMM gradient computation
        let conv = Conv2d::new(2, 4, 3, 1, 1, true);
        conv.set_algo(ConvAlgo::IGEMM);

        let x = RawTensor::randn(&[1, 2, 8, 8]);
        x.borrow_mut().requires_grad = true;

        let (max_err, mean_err, passed) =
            RawTensor::check_gradients(&x, |t| conv.forward(t).sum(), 5e-2, 2e-2);

        assert!(
            passed,
            "iGEMM gradient check failed: max_error={max_err:.6e}, mean_error={mean_err:.6e}"
        );
    }

    #[test]
    fn test_conv2d_igemm_gradient_match_direct() {
        // Compare iGEMM gradients with Direct gradients
        // Use SAME input values for fair comparison
        let x_direct = RawTensor::randn(&[1, 2, 6, 6]);
        x_direct.borrow_mut().requires_grad = true;

        // Clone input data to use same values
        let x_data = x_direct.borrow().data.to_vec();
        let x_igemm = RawTensor::new(x_data, &[1, 2, 6, 6], true);
        x_igemm.borrow_mut().requires_grad = true;

        let conv_direct = Conv2d::new(2, 4, 3, 1, 1, true);
        conv_direct.set_algo(ConvAlgo::Direct);

        let conv_igemm = Conv2d::new(2, 4, 3, 1, 1, true);
        conv_igemm.set_algo(ConvAlgo::IGEMM);

        // Copy weights
        let weight_data = conv_direct.weight.borrow().data.to_vec();
        conv_igemm.weight.borrow_mut().data = crate::storage::Storage::cpu(weight_data.clone());

        let bias_data = conv_direct.bias.as_ref().unwrap().borrow().data.to_vec();
        conv_igemm.bias.as_ref().unwrap().borrow_mut().data =
            crate::storage::Storage::cpu(bias_data.clone());

        // Forward and backward
        let y_direct = conv_direct.forward(&x_direct);
        let loss_direct = y_direct.sum();
        loss_direct.backward();

        let y_igemm = conv_igemm.forward(&x_igemm);
        let loss_igemm = y_igemm.sum();
        loss_igemm.backward();

        // Compare input gradients
        let grad_direct = x_direct.grad().unwrap();
        let grad_igemm = x_igemm.grad().unwrap();

        assert_eq!(grad_direct.len(), grad_igemm.len());
        for (i, (direct_val, igemm_val)) in grad_direct.iter().zip(grad_igemm.iter()).enumerate() {
            let abs_diff = (direct_val - igemm_val).abs();
            assert!(
                abs_diff < 1e-4,
                "Input grad mismatch at index {i}: Direct={direct_val}, IGEMM={igemm_val}, diff={abs_diff}"
            );
        }

        // Compare weight gradients
        let w_grad_direct = conv_direct.weight.grad().unwrap();
        let w_grad_igemm = conv_igemm.weight.grad().unwrap();

        assert_eq!(w_grad_direct.len(), w_grad_igemm.len());
        for (i, (direct_val, igemm_val)) in
            w_grad_direct.iter().zip(w_grad_igemm.iter()).enumerate()
        {
            let abs_diff = (direct_val - igemm_val).abs();
            assert!(
                abs_diff < 1e-4,
                "Weight grad mismatch at index {i}: Direct={direct_val}, IGEMM={igemm_val}, diff={abs_diff}"
            );
        }
    }

    #[test]
    fn test_conv2d_igemm_parameter_variants() {
        // Test iGEMM with various parameter combinations
        let test_cases = vec![
            (3, 1, 1, 2, 4, 8, 8, 8), // kernel, stride, padding, batch, in_ch, out_ch, h, w
            (5, 2, 2, 1, 3, 6, 12, 12),
            (1, 1, 0, 4, 4, 8, 8, 8), // 1x1 kernel
            (7, 1, 3, 1, 2, 4, 16, 16),
        ];

        for (kernel, stride, padding, batch, in_ch, out_ch, h, w) in test_cases {
            let conv = Conv2d::new(in_ch, out_ch, kernel, stride, padding, true);
            conv.set_algo(ConvAlgo::IGEMM);

            let x = RawTensor::randn(&[batch, in_ch, h, w]);
            let y = conv.forward(&x);

            // Just verify it runs without crashing and produces correct shape
            let h_out = (h + 2 * padding - kernel) / stride + 1;
            let w_out = (w + 2 * padding - kernel) / stride + 1;
            assert_eq!(y.borrow().shape, vec![batch, out_ch, h_out, w_out]);
        }
    }

    #[test]
    fn test_conv2d_igemm_no_bias() {
        // Test iGEMM without bias
        let conv = Conv2d::new(2, 4, 3, 1, 1, false);
        conv.set_algo(ConvAlgo::IGEMM);

        let x = RawTensor::randn(&[1, 2, 8, 8]);
        let y = conv.forward(&x);

        assert_eq!(y.borrow().shape, vec![1, 4, 8, 8]);
        assert!(conv.bias.is_none());
    }

    #[test]
    fn test_conv2d_igemm_gradient_no_bias() {
        // Test iGEMM gradient without bias
        let conv = Conv2d::new(2, 4, 3, 1, 1, false);
        conv.set_algo(ConvAlgo::IGEMM);

        let x = RawTensor::randn(&[1, 2, 8, 8]);
        x.borrow_mut().requires_grad = true;

        let (max_err, mean_err, passed) =
            RawTensor::check_gradients(&x, |t| conv.forward(t).sum(), 5e-2, 2e-2);

        assert!(
            passed,
            "iGEMM no-bias gradient check failed: max_error={max_err:.6e}, mean_error={mean_err:.6e}"
        );
    }
}

#[cfg(all(test, feature = "gpu"))]
mod conv2d_gpu_tests {
    use super::*;
    use crate::Device;

    #[test]
    fn test_conv2d_gpu_forward_shape() {
        if Device::gpu().is_none() {
            return; // Skip if no GPU available
        }

        let device = Device::gpu().unwrap();

        // Create Conv2d layer on GPU
        let conv = Conv2d::new_on_device(3, 16, 3, 1, 1, true, device.clone());
        let x = RawTensor::randn(&[1, 3, 32, 32]).to_device(device.clone());
        let y = conv.forward(&x);

        assert_eq!(y.borrow().shape, vec![1, 16, 32, 32]);
        assert!(y.borrow().device.is_gpu(), "Output should be on GPU");
    }

    #[test]
    fn test_conv2d_gpu_forward_match_cpu() {
        if Device::gpu().is_none() {
            return;
        }

        let device = Device::gpu().unwrap();

        // Create CPU conv with deterministic weights (all ones)
        let conv_cpu = Conv2d::new_on_device(2, 4, 3, 1, 1, true, Device::CPU);
        // Set weights to all ones
        let weight_size = 2 * 4 * 3 * 3; // in_channels * out_channels * kernel_h * kernel_w
        conv_cpu.weight.borrow_mut().data = crate::storage::Storage::cpu(vec![1.0; weight_size]);
        // Set bias to zeros
        conv_cpu.bias.as_ref().unwrap().borrow_mut().data =
            crate::storage::Storage::cpu(vec![0.0; 4]);

        // Create deterministic input (all ones)
        let x_cpu = RawTensor::new(vec![1.0; 128], &[1, 2, 8, 8], false);
        let y_cpu = conv_cpu.forward(&x_cpu);

        // Move entire conv layer to GPU
        let conv_gpu = Conv2d::new_on_device(2, 4, 3, 1, 1, true, device.clone());
        conv_gpu.weight.borrow_mut().data = crate::storage::Storage::gpu(vec![1.0; weight_size]);
        conv_gpu.bias.as_ref().unwrap().borrow_mut().data =
            crate::storage::Storage::gpu(vec![0.0; 4]);

        let x_gpu = x_cpu.to_device(device.clone());
        let y_gpu = conv_gpu.forward(&x_gpu);

        // Compare shapes
        assert_eq!(y_cpu.borrow().shape, y_gpu.borrow().shape);

        // Compare values (allow for floating point differences)
        let y_cpu_data = y_cpu.borrow().data.to_vec();
        let y_gpu_data = y_gpu.borrow().data.to_vec();

        assert_eq!(y_cpu_data.len(), y_gpu_data.len());

        for (i, (cpu_val, gpu_val)) in y_cpu_data.iter().zip(y_gpu_data.iter()).enumerate() {
            let abs_diff = (cpu_val - gpu_val).abs();
            assert!(
                abs_diff < 1e-4,
                "Mismatch at index {i}: CPU={cpu_val}, GPU={gpu_val}, abs_diff={abs_diff}"
            );
        }
    }

    #[test]
    fn test_conv2d_gpu_backward_flow() {
        if Device::gpu().is_none() {
            return;
        }

        let device = Device::gpu().unwrap();

        let conv = Conv2d::new_on_device(2, 4, 3, 1, 0, true, device.clone());
        let x = RawTensor::randn(&[1, 2, 5, 5]).to_device(device.clone());
        x.borrow_mut().requires_grad = true;

        let y = conv.forward(&x);
        let loss = y.sum();
        loss.backward();

        // Check that gradients exist
        assert!(x.grad().is_some(), "Input should have gradients");
        assert!(
            conv.weight.grad().is_some(),
            "Weights should have gradients"
        );
        if let Some(ref b) = conv.bias {
            assert!(b.grad().is_some(), "Bias should have gradients");
        }
    }

    #[test]
    fn test_conv2d_gpu_stride2() {
        if Device::gpu().is_none() {
            return;
        }

        let device = Device::gpu().unwrap();

        let conv = Conv2d::new_on_device(1, 4, 3, 2, 1, true, device.clone());
        let x = RawTensor::randn(&[1, 1, 8, 8]).to_device(device.clone());
        let y = conv.forward(&x);

        assert_eq!(y.borrow().shape, vec![1, 4, 4, 4]);
        assert!(y.borrow().device.is_gpu(), "Output should be on GPU");
    }

    #[test]
    fn test_conv2d_gpu_simple_values() {
        if Device::gpu().is_none() {
            return;
        }

        let device = Device::gpu().unwrap();

        // Create simple input with known values
        // Input: 1x1x4x4 with values 0-15
        let x_cpu = RawTensor::new(
            (0..16).map(|i| i as f32).collect::<Vec<_>>(),
            &[1, 1, 4, 4],
            false,
        );

        // Create Conv2d with simple weights (1 output channel, 1 input channel, 2x2 kernel, stride 1, padding 0)
        // Weights: all ones
        let conv_cpu = Conv2d::new_on_device(1, 1, 2, 1, 0, false, Device::CPU);
        conv_cpu.weight.borrow_mut().data = crate::storage::Storage::cpu(vec![1.0; 4]);

        let y_cpu = conv_cpu.forward(&x_cpu);

        // Now try on GPU
        let x_gpu = x_cpu.to_device(device.clone());
        let conv_gpu = Conv2d::new_on_device(1, 1, 2, 1, 0, false, device.clone());
        conv_gpu.weight.borrow_mut().data = crate::storage::Storage::gpu(vec![1.0; 4]);

        let y_gpu = conv_gpu.forward(&x_gpu);

        // Compare shapes
        assert_eq!(y_cpu.borrow().shape, y_gpu.borrow().shape);

        // Compare values
        let y_cpu_data = y_cpu.borrow().data.to_vec();
        let y_gpu_data = y_gpu.borrow().data.to_vec();

        for (i, (cpu_val, gpu_val)) in y_cpu_data.iter().zip(y_gpu_data.iter()).enumerate() {
            let abs_diff = (cpu_val - gpu_val).abs();
            assert!(
                abs_diff < 1e-4,
                "Mismatch at index {i}: CPU={cpu_val}, GPU={gpu_val}, abs_diff={abs_diff}"
            );
        }
    }

    #[test]
    fn test_conv2d_gpu_padding() {
        if Device::gpu().is_none() {
            return;
        }

        let device = Device::gpu().unwrap();

        // Test with padding to ensure pad operation works on GPU
        let x_cpu = RawTensor::new(vec![1.0; 16], &[1, 1, 4, 4], false);
        let x_padded_cpu = x_cpu.pad(&[(0, 0), (0, 0), (1, 1), (1, 1)]);

        let x_gpu = x_cpu.to_device(device.clone());
        let x_padded_gpu = x_gpu.pad(&[(0, 0), (0, 0), (1, 1), (1, 1)]);

        // Check shapes
        assert_eq!(x_padded_cpu.borrow().shape, vec![1, 1, 6, 6]);
        assert_eq!(x_padded_gpu.borrow().shape, vec![1, 1, 6, 6]);

        // Check values - center should be 1.0, edges should be 0.0
        let padded_cpu_data = x_padded_cpu.borrow().data.to_vec();
        let padded_gpu_data = x_padded_gpu.borrow().data.to_vec();

        for (i, (cpu_val, gpu_val)) in padded_cpu_data
            .iter()
            .zip(padded_gpu_data.iter())
            .enumerate()
        {
            let abs_diff = (cpu_val - gpu_val).abs();
            assert!(
                abs_diff < 1e-4,
                "Padding mismatch at index {i}: CPU={cpu_val}, GPU={gpu_val}"
            );
        }
    }

    // ===== Comprehensive CPU-GPU Consistency Tests =====

    #[test]
    fn test_conv2d_gpu_cpu_consistency_1x1_kernel() {
        if Device::gpu().is_none() {
            return;
        }

        let device = Device::gpu().unwrap();

        // Test 1x1 kernel consistency with deterministic values
        let x_cpu = RawTensor::new(vec![1.0; 256], &[1, 4, 8, 8], false);

        // Create CPU conv with deterministic weights
        let conv_cpu = Conv2d::new_on_device(4, 8, 1, 1, 0, true, Device::CPU);
        #[allow(clippy::identity_op)]
        let weight_size = 4 * 8 * 1 * 1;
        conv_cpu.weight.borrow_mut().data = crate::storage::Storage::cpu(vec![1.0; weight_size]);
        conv_cpu.bias.as_ref().unwrap().borrow_mut().data =
            crate::storage::Storage::cpu(vec![0.0; 8]);

        let y_cpu = conv_cpu.forward(&x_cpu);

        // Create GPU conv with same weights
        let x_gpu = x_cpu.to_device(device.clone());
        let conv_gpu = Conv2d::new_on_device(4, 8, 1, 1, 0, true, device.clone());
        conv_gpu.weight.borrow_mut().data = crate::storage::Storage::gpu(vec![1.0; weight_size]);
        conv_gpu.bias.as_ref().unwrap().borrow_mut().data =
            crate::storage::Storage::gpu(vec![0.0; 8]);

        let y_gpu = conv_gpu.forward(&x_gpu);

        assert_eq!(y_cpu.borrow().shape, y_gpu.borrow().shape);

        let y_cpu_data = y_cpu.borrow().data.to_vec();
        let y_gpu_data = y_gpu.borrow().data.to_vec();

        for (i, (cpu_val, gpu_val)) in y_cpu_data.iter().zip(y_gpu_data.iter()).enumerate() {
            let abs_diff = (cpu_val - gpu_val).abs();
            assert!(
                abs_diff < 1e-4,
                "1x1 kernel mismatch at index {i}: CPU={cpu_val}, GPU={gpu_val}, diff={abs_diff}"
            );
        }
    }

    #[test]
    fn test_conv2d_gpu_cpu_consistency_5x5_kernel() {
        if Device::gpu().is_none() {
            return;
        }

        let device = Device::gpu().unwrap();

        // Test 5x5 kernel consistency
        let x_cpu = RawTensor::new(vec![1.0; 768], &[1, 3, 16, 16], false);

        let conv_cpu = Conv2d::new_on_device(3, 8, 5, 1, 2, true, Device::CPU);
        let weight_size = 3 * 8 * 5 * 5;
        conv_cpu.weight.borrow_mut().data = crate::storage::Storage::cpu(vec![1.0; weight_size]);
        conv_cpu.bias.as_ref().unwrap().borrow_mut().data =
            crate::storage::Storage::cpu(vec![0.0; 8]);

        let y_cpu = conv_cpu.forward(&x_cpu);

        let x_gpu = x_cpu.to_device(device.clone());
        let conv_gpu = Conv2d::new_on_device(3, 8, 5, 1, 2, true, device.clone());
        conv_gpu.weight.borrow_mut().data = crate::storage::Storage::gpu(vec![1.0; weight_size]);
        conv_gpu.bias.as_ref().unwrap().borrow_mut().data =
            crate::storage::Storage::gpu(vec![0.0; 8]);

        let y_gpu = conv_gpu.forward(&x_gpu);

        assert_eq!(y_cpu.borrow().shape, y_gpu.borrow().shape);

        let y_cpu_data = y_cpu.borrow().data.to_vec();
        let y_gpu_data = y_gpu.borrow().data.to_vec();

        for (i, (cpu_val, gpu_val)) in y_cpu_data.iter().zip(y_gpu_data.iter()).enumerate() {
            let abs_diff = (cpu_val - gpu_val).abs();
            assert!(
                abs_diff < 1e-4,
                "5x5 kernel mismatch at index {i}: CPU={cpu_val}, GPU={gpu_val}, diff={abs_diff}"
            );
        }
    }

    #[test]
    fn test_conv2d_gpu_cpu_consistency_stride2() {
        if Device::gpu().is_none() {
            return;
        }

        let device = Device::gpu().unwrap();

        // Test stride 2 consistency
        let x_cpu = RawTensor::new(vec![1.0; 768], &[1, 3, 16, 16], false);

        let conv_cpu = Conv2d::new_on_device(3, 8, 3, 2, 1, true, Device::CPU);
        let weight_size = 3 * 8 * 3 * 3;
        conv_cpu.weight.borrow_mut().data = crate::storage::Storage::cpu(vec![1.0; weight_size]);
        conv_cpu.bias.as_ref().unwrap().borrow_mut().data =
            crate::storage::Storage::cpu(vec![0.0; 8]);

        let y_cpu = conv_cpu.forward(&x_cpu);

        let x_gpu = x_cpu.to_device(device.clone());
        let conv_gpu = Conv2d::new_on_device(3, 8, 3, 2, 1, true, device.clone());
        conv_gpu.weight.borrow_mut().data = crate::storage::Storage::gpu(vec![1.0; weight_size]);
        conv_gpu.bias.as_ref().unwrap().borrow_mut().data =
            crate::storage::Storage::gpu(vec![0.0; 8]);

        let y_gpu = conv_gpu.forward(&x_gpu);

        assert_eq!(y_cpu.borrow().shape, y_gpu.borrow().shape);

        let y_cpu_data = y_cpu.borrow().data.to_vec();
        let y_gpu_data = y_gpu.borrow().data.to_vec();

        for (i, (cpu_val, gpu_val)) in y_cpu_data.iter().zip(y_gpu_data.iter()).enumerate() {
            let abs_diff = (cpu_val - gpu_val).abs();
            assert!(
                abs_diff < 1e-4,
                "Stride2 mismatch at index {i}: CPU={cpu_val}, GPU={gpu_val}, diff={abs_diff}"
            );
        }
    }

    #[test]
    fn test_conv2d_gpu_cpu_consistency_stride4() {
        if Device::gpu().is_none() {
            return;
        }

        let device = Device::gpu().unwrap();

        // Test stride 4 consistency
        let x_cpu = RawTensor::new(vec![1.0; 3072], &[1, 3, 32, 32], false);

        let conv_cpu = Conv2d::new_on_device(3, 8, 3, 4, 1, true, Device::CPU);
        let weight_size = 3 * 8 * 3 * 3;
        conv_cpu.weight.borrow_mut().data = crate::storage::Storage::cpu(vec![1.0; weight_size]);
        conv_cpu.bias.as_ref().unwrap().borrow_mut().data =
            crate::storage::Storage::cpu(vec![0.0; 8]);

        let y_cpu = conv_cpu.forward(&x_cpu);

        let x_gpu = x_cpu.to_device(device.clone());
        let conv_gpu = Conv2d::new_on_device(3, 8, 3, 4, 1, true, device.clone());
        conv_gpu.weight.borrow_mut().data = crate::storage::Storage::gpu(vec![1.0; weight_size]);
        conv_gpu.bias.as_ref().unwrap().borrow_mut().data =
            crate::storage::Storage::gpu(vec![0.0; 8]);

        let y_gpu = conv_gpu.forward(&x_gpu);

        assert_eq!(y_cpu.borrow().shape, y_gpu.borrow().shape);

        let y_cpu_data = y_cpu.borrow().data.to_vec();
        let y_gpu_data = y_gpu.borrow().data.to_vec();

        for (i, (cpu_val, gpu_val)) in y_cpu_data.iter().zip(y_gpu_data.iter()).enumerate() {
            let abs_diff = (cpu_val - gpu_val).abs();
            assert!(
                abs_diff < 1e-4,
                "Stride4 mismatch at index {i}: CPU={cpu_val}, GPU={gpu_val}, diff={abs_diff}"
            );
        }
    }

    #[test]
    fn test_conv2d_gpu_cpu_consistency_large_padding() {
        if Device::gpu().is_none() {
            return;
        }

        let device = Device::gpu().unwrap();

        // Test large padding consistency
        let x_cpu = RawTensor::new(vec![1.0; 192], &[1, 3, 8, 8], false);

        let conv_cpu = Conv2d::new_on_device(3, 8, 3, 1, 3, true, Device::CPU);
        let weight_size = 3 * 8 * 3 * 3;
        conv_cpu.weight.borrow_mut().data = crate::storage::Storage::cpu(vec![1.0; weight_size]);
        conv_cpu.bias.as_ref().unwrap().borrow_mut().data =
            crate::storage::Storage::cpu(vec![0.0; 8]);

        let y_cpu = conv_cpu.forward(&x_cpu);

        let x_gpu = x_cpu.to_device(device.clone());
        let conv_gpu = Conv2d::new_on_device(3, 8, 3, 1, 3, true, device.clone());
        conv_gpu.weight.borrow_mut().data = crate::storage::Storage::gpu(vec![1.0; weight_size]);
        conv_gpu.bias.as_ref().unwrap().borrow_mut().data =
            crate::storage::Storage::gpu(vec![0.0; 8]);

        let y_gpu = conv_gpu.forward(&x_gpu);

        assert_eq!(y_cpu.borrow().shape, y_gpu.borrow().shape);

        let y_cpu_data = y_cpu.borrow().data.to_vec();
        let y_gpu_data = y_gpu.borrow().data.to_vec();

        for (i, (cpu_val, gpu_val)) in y_cpu_data.iter().zip(y_gpu_data.iter()).enumerate() {
            let abs_diff = (cpu_val - gpu_val).abs();
            assert!(
                abs_diff < 1e-4,
                "Large padding mismatch at index {i}: CPU={cpu_val}, GPU={gpu_val}, diff={abs_diff}"
            );
        }
    }

    #[test]
    fn test_conv2d_gpu_cpu_consistency_no_bias() {
        if Device::gpu().is_none() {
            return;
        }

        let device = Device::gpu().unwrap();

        // Test without bias
        let x_cpu = RawTensor::new(vec![1.0; 768], &[1, 3, 16, 16], false);

        let conv_cpu = Conv2d::new_on_device(3, 8, 3, 1, 1, false, Device::CPU);
        let weight_size = 3 * 8 * 3 * 3;
        conv_cpu.weight.borrow_mut().data = crate::storage::Storage::cpu(vec![1.0; weight_size]);

        let y_cpu = conv_cpu.forward(&x_cpu);

        let x_gpu = x_cpu.to_device(device.clone());
        let conv_gpu = Conv2d::new_on_device(3, 8, 3, 1, 1, false, device.clone());
        conv_gpu.weight.borrow_mut().data = crate::storage::Storage::gpu(vec![1.0; weight_size]);

        let y_gpu = conv_gpu.forward(&x_gpu);

        assert_eq!(y_cpu.borrow().shape, y_gpu.borrow().shape);

        let y_cpu_data = y_cpu.borrow().data.to_vec();
        let y_gpu_data = y_gpu.borrow().data.to_vec();

        for (i, (cpu_val, gpu_val)) in y_cpu_data.iter().zip(y_gpu_data.iter()).enumerate() {
            let abs_diff = (cpu_val - gpu_val).abs();
            assert!(
                abs_diff < 1e-4,
                "No bias mismatch at index {i}: CPU={cpu_val}, GPU={gpu_val}, diff={abs_diff}"
            );
        }
    }

    #[test]
    fn test_conv2d_gpu_cpu_consistency_batch_processing() {
        if Device::gpu().is_none() {
            return;
        }

        let device = Device::gpu().unwrap();

        // Test different batch sizes
        for batch in [1, 2, 4, 8] {
            let size = batch * 3 * 16 * 16;
            let x_cpu = RawTensor::new(vec![1.0; size], &[batch, 3, 16, 16], false);

            let conv_cpu = Conv2d::new_on_device(3, 8, 3, 1, 1, true, Device::CPU);
            let weight_size = 3 * 8 * 3 * 3;
            conv_cpu.weight.borrow_mut().data =
                crate::storage::Storage::cpu(vec![1.0; weight_size]);
            conv_cpu.bias.as_ref().unwrap().borrow_mut().data =
                crate::storage::Storage::cpu(vec![0.0; 8]);

            let y_cpu = conv_cpu.forward(&x_cpu);

            let x_gpu = x_cpu.to_device(device.clone());
            let conv_gpu = Conv2d::new_on_device(3, 8, 3, 1, 1, true, device.clone());
            conv_gpu.weight.borrow_mut().data =
                crate::storage::Storage::gpu(vec![1.0; weight_size]);
            conv_gpu.bias.as_ref().unwrap().borrow_mut().data =
                crate::storage::Storage::gpu(vec![0.0; 8]);

            let y_gpu = conv_gpu.forward(&x_gpu);

            assert_eq!(y_cpu.borrow().shape, y_gpu.borrow().shape);

            let y_cpu_data = y_cpu.borrow().data.to_vec();
            let y_gpu_data = y_gpu.borrow().data.to_vec();

            for (i, (cpu_val, gpu_val)) in y_cpu_data.iter().zip(y_gpu_data.iter()).enumerate() {
                let abs_diff = (cpu_val - gpu_val).abs();
                assert!(
                    abs_diff < 1e-4,
                    "Batch {batch} mismatch at index {i}: CPU={cpu_val}, GPU={gpu_val}, diff={abs_diff}"
                );
            }
        }
    }

    #[test]
    fn test_conv2d_gpu_cpu_consistency_asymmetric_dimensions() {
        if Device::gpu().is_none() {
            return;
        }

        let device = Device::gpu().unwrap();

        // Test non-square spatial dimensions
        let x_cpu = RawTensor::new(vec![1.0; 1536], &[1, 3, 16, 32], false);

        let conv_cpu = Conv2d::new_on_device(3, 8, 3, 2, 1, true, Device::CPU);
        let weight_size = 3 * 8 * 3 * 3;
        conv_cpu.weight.borrow_mut().data = crate::storage::Storage::cpu(vec![1.0; weight_size]);
        conv_cpu.bias.as_ref().unwrap().borrow_mut().data =
            crate::storage::Storage::cpu(vec![0.0; 8]);

        let y_cpu = conv_cpu.forward(&x_cpu);

        let x_gpu = x_cpu.to_device(device.clone());
        let conv_gpu = Conv2d::new_on_device(3, 8, 3, 2, 1, true, device.clone());
        conv_gpu.weight.borrow_mut().data = crate::storage::Storage::gpu(vec![1.0; weight_size]);
        conv_gpu.bias.as_ref().unwrap().borrow_mut().data =
            crate::storage::Storage::gpu(vec![0.0; 8]);

        let y_gpu = conv_gpu.forward(&x_gpu);

        assert_eq!(y_cpu.borrow().shape, y_gpu.borrow().shape);

        let y_cpu_data = y_cpu.borrow().data.to_vec();
        let y_gpu_data = y_gpu.borrow().data.to_vec();

        for (i, (cpu_val, gpu_val)) in y_cpu_data.iter().zip(y_gpu_data.iter()).enumerate() {
            let abs_diff = (cpu_val - gpu_val).abs();
            assert!(
                abs_diff < 1e-4,
                "Asymmetric dimensions mismatch at index {i}: CPU={cpu_val}, GPU={gpu_val}, diff={abs_diff}"
            );
        }
    }

    #[test]
    fn test_conv2d_gpu_cpu_consistency_multiple_channels() {
        if Device::gpu().is_none() {
            return;
        }

        let device = Device::gpu().unwrap();

        // Test various channel configurations
        let test_cases = vec![(1, 1, 8, 8, 64), (3, 16, 16, 16, 768), (16, 32, 8, 8, 1024)];

        for (in_ch, out_ch, h, w, size) in test_cases {
            let x_cpu = RawTensor::new(vec![1.0; size], &[1, in_ch, h, w], false);

            let conv_cpu = Conv2d::new_on_device(in_ch, out_ch, 3, 1, 1, true, Device::CPU);
            let weight_size = in_ch * out_ch * 3 * 3;
            conv_cpu.weight.borrow_mut().data =
                crate::storage::Storage::cpu(vec![1.0; weight_size]);
            conv_cpu.bias.as_ref().unwrap().borrow_mut().data =
                crate::storage::Storage::cpu(vec![0.0; out_ch]);

            let y_cpu = conv_cpu.forward(&x_cpu);

            let x_gpu = x_cpu.to_device(device.clone());
            let conv_gpu = Conv2d::new_on_device(in_ch, out_ch, 3, 1, 1, true, device.clone());
            conv_gpu.weight.borrow_mut().data =
                crate::storage::Storage::gpu(vec![1.0; weight_size]);
            conv_gpu.bias.as_ref().unwrap().borrow_mut().data =
                crate::storage::Storage::gpu(vec![0.0; out_ch]);

            let y_gpu = conv_gpu.forward(&x_gpu);

            assert_eq!(y_cpu.borrow().shape, y_gpu.borrow().shape);

            let y_cpu_data = y_cpu.borrow().data.to_vec();
            let y_gpu_data = y_gpu.borrow().data.to_vec();

            for (i, (cpu_val, gpu_val)) in y_cpu_data.iter().zip(y_gpu_data.iter()).enumerate() {
                let abs_diff = (cpu_val - gpu_val).abs();
                assert!(
                    abs_diff < 1e-4,
                    "Channel configuration ({in_ch},{out_ch}) mismatch at index {i}: CPU={cpu_val}, GPU={gpu_val}, diff={abs_diff}"
                );
            }
        }
    }

    #[test]
    fn test_conv2d_gpu_gradient_flow_consistency() {
        if Device::gpu().is_none() {
            return;
        }

        let device = Device::gpu().unwrap();

        // Test that gradients flow correctly on GPU
        let conv = Conv2d::new_on_device(2, 4, 3, 1, 1, true, device.clone());
        let x = RawTensor::randn(&[1, 2, 6, 6]).to_device(device.clone());
        x.borrow_mut().requires_grad = true;

        let y = conv.forward(&x);
        let loss = y.sum();
        loss.backward();

        // Verify all gradients exist
        assert!(x.grad().is_some(), "Input should have gradients");
        assert!(
            conv.weight.grad().is_some(),
            "Weights should have gradients"
        );
        assert!(
            conv.bias.as_ref().unwrap().grad().is_some(),
            "Bias should have gradients"
        );

        // Verify gradient shapes
        let x_grad = x.grad().unwrap();
        // 1 x 2 x 6 x 6
        assert_eq!(x_grad.len(), 2 * 6 * 6, "Input grad shape mismatch");

        let w_grad = conv.weight.grad().unwrap();
        assert_eq!(w_grad.len(), 2 * 4 * 3 * 3, "Weight grad shape mismatch");

        let b_grad = conv.bias.as_ref().unwrap().grad().unwrap();
        assert_eq!(b_grad.len(), 4, "Bias grad shape mismatch");
    }

    #[test]
    fn test_conv2d_gpu_direct_conv_cpu_consistency() {
        if Device::gpu().is_none() {
            return;
        }

        let device = Device::gpu().unwrap();

        // Test various kernel sizes
        for kernel_size in [1, 3, 5, 7] {
            let x_cpu = RawTensor::new(vec![1.0; 768], &[1, 3, 16, 16], false);

            let conv_cpu =
                Conv2d::new_on_device(3, 8, kernel_size, 1, kernel_size / 2, true, Device::CPU);
            let weight_size = 3 * 8 * kernel_size * kernel_size;
            conv_cpu.weight.borrow_mut().data =
                crate::storage::Storage::cpu(vec![1.0; weight_size]);
            conv_cpu.bias.as_ref().unwrap().borrow_mut().data =
                crate::storage::Storage::cpu(vec![0.0; 8]);

            conv_cpu.set_algo(ConvAlgo::Direct);
            let y_cpu = conv_cpu.forward(&x_cpu);

            // GPU version
            let x_gpu = x_cpu.to_device(device.clone());
            let conv_gpu =
                Conv2d::new_on_device(3, 8, kernel_size, 1, kernel_size / 2, true, device.clone());
            conv_gpu.weight.borrow_mut().data =
                crate::storage::Storage::gpu(vec![1.0; weight_size]);
            conv_gpu.bias.as_ref().unwrap().borrow_mut().data =
                crate::storage::Storage::gpu(vec![0.0; 8]);
            conv_gpu.set_algo(ConvAlgo::Direct);

            let y_gpu = conv_gpu.forward(&x_gpu);

            // Compare results
            let y_cpu_data = y_cpu.borrow().data.to_vec();
            let y_gpu_data = y_gpu.borrow().data.to_vec();

            assert_eq!(y_cpu_data.len(), y_gpu_data.len());
            for (i, (cpu_val, gpu_val)) in y_cpu_data.iter().zip(y_gpu_data.iter()).enumerate() {
                let abs_diff = (cpu_val - gpu_val).abs();
                assert!(
                    abs_diff < 1e-4,
                    "Kernel {kernel_size} mismatch at index {i}: CPU={cpu_val}, GPU={gpu_val}, diff={abs_diff}"
                );
            }
        }
    }

    #[test]
    fn test_conv2d_gpu_direct_conv_parameter_variants() {
        if Device::gpu().is_none() {
            return;
        }

        let device = Device::gpu().unwrap();

        // Test different parameter combinations
        let test_cases = vec![
            // (kernel, stride, padding, batch, in_ch, out_ch, h, w)
            (3, 1, 1, 1, 2, 4, 8, 8),
            (3, 2, 1, 2, 3, 6, 16, 16),
            (5, 1, 2, 1, 4, 8, 12, 12),
            (5, 2, 2, 4, 3, 6, 24, 24),
            (7, 1, 3, 1, 2, 4, 16, 16),
            (1, 1, 0, 8, 4, 8, 8, 8), // 1x1 kernel
        ];

        for (kernel, stride, padding, batch, in_ch, out_ch, h, w) in test_cases {
            let x_cpu = RawTensor::new(
                vec![1.0; batch * in_ch * h * w],
                &[batch, in_ch, h, w],
                false,
            );

            let conv_cpu =
                Conv2d::new_on_device(in_ch, out_ch, kernel, stride, padding, true, Device::CPU);
            let weight_size = in_ch * out_ch * kernel * kernel;
            conv_cpu.weight.borrow_mut().data =
                crate::storage::Storage::cpu(vec![1.0; weight_size]);
            conv_cpu.bias.as_ref().unwrap().borrow_mut().data =
                crate::storage::Storage::cpu(vec![0.0; out_ch]);

            conv_cpu.set_algo(ConvAlgo::Direct);
            let y_cpu = conv_cpu.forward(&x_cpu);

            // GPU version
            let x_gpu = x_cpu.to_device(device.clone());
            let conv_gpu =
                Conv2d::new_on_device(in_ch, out_ch, kernel, stride, padding, true, device.clone());
            conv_gpu.weight.borrow_mut().data =
                crate::storage::Storage::gpu(vec![1.0; weight_size]);
            conv_gpu.bias.as_ref().unwrap().borrow_mut().data =
                crate::storage::Storage::gpu(vec![0.0; out_ch]);
            conv_gpu.set_algo(ConvAlgo::Direct);

            let y_gpu = conv_gpu.forward(&x_gpu);

            // Compare results
            let y_cpu_data = y_cpu.borrow().data.to_vec();
            let y_gpu_data = y_gpu.borrow().data.to_vec();

            assert_eq!(y_cpu_data.len(), y_gpu_data.len());
            for (i, (cpu_val, gpu_val)) in y_cpu_data.iter().zip(y_gpu_data.iter()).enumerate() {
                let abs_diff = (cpu_val - gpu_val).abs();
                assert!(
                    abs_diff < 1e-4,
                    "Params k={kernel} s={stride} p={padding} b={batch} mismatch at {i}: CPU={cpu_val}, GPU={gpu_val}, diff={abs_diff}"
                );
            }
        }
    }

    #[test]
    fn test_conv2d_gpu_direct_conv_backward_flow() {
        if Device::gpu().is_none() {
            return;
        }

        let device = Device::gpu().unwrap();

        let conv = Conv2d::new_on_device(2, 4, 3, 1, 1, true, device.clone());
        conv.set_algo(ConvAlgo::Direct);

        let x = RawTensor::randn(&[1, 2, 8, 8]).to_device(device.clone());
        x.borrow_mut().requires_grad = true;

        let y = conv.forward(&x);
        let loss = y.sum();
        loss.backward();

        // Gradients should exist (computed on CPU after GPU forward)
        assert!(x.grad().is_some(), "Input should have gradients");
        assert!(
            conv.weight.grad().is_some(),
            "Weights should have gradients"
        );
        assert!(
            conv.bias.as_ref().unwrap().grad().is_some(),
            "Bias should have gradients"
        );
    }

    #[test]
    fn test_conv2d_gpu_direct_conv_non_square() {
        if Device::gpu().is_none() {
            return;
        }

        let device = Device::gpu().unwrap();

        // Test non-square spatial dimensions
        let x_cpu = RawTensor::new(vec![1.0; 1536], &[1, 3, 16, 32], false);

        let conv_cpu = Conv2d::new_on_device(3, 8, 3, 1, 1, true, Device::CPU);
        let weight_size = 3 * 8 * 3 * 3;
        conv_cpu.weight.borrow_mut().data = crate::storage::Storage::cpu(vec![1.0; weight_size]);
        conv_cpu.bias.as_ref().unwrap().borrow_mut().data =
            crate::storage::Storage::cpu(vec![0.0; 8]);

        conv_cpu.set_algo(ConvAlgo::Direct);
        let y_cpu = conv_cpu.forward(&x_cpu);

        // GPU version
        let x_gpu = x_cpu.to_device(device.clone());
        let conv_gpu = Conv2d::new_on_device(3, 8, 3, 1, 1, true, device.clone());
        conv_gpu.weight.borrow_mut().data = crate::storage::Storage::gpu(vec![1.0; weight_size]);
        conv_gpu.bias.as_ref().unwrap().borrow_mut().data =
            crate::storage::Storage::gpu(vec![0.0; 8]);
        conv_gpu.set_algo(ConvAlgo::Direct);

        let y_gpu = conv_gpu.forward(&x_gpu);

        // Compare results
        let y_cpu_data = y_cpu.borrow().data.to_vec();
        let y_gpu_data = y_gpu.borrow().data.to_vec();

        assert_eq!(y_cpu_data.len(), y_gpu_data.len());
        for (i, (cpu_val, gpu_val)) in y_cpu_data.iter().zip(y_gpu_data.iter()).enumerate() {
            let abs_diff = (cpu_val - gpu_val).abs();
            assert!(
                abs_diff < 1e-4,
                "Non-square mismatch at index {i}: CPU={cpu_val}, GPU={gpu_val}, diff={abs_diff}"
            );
        }
    }

    // ===== GPU col2im Tests =====

    #[test]
    fn test_conv2d_gpu_col2im_cpu_consistency() {
        if Device::gpu().is_none() {
            return;
        }

        let device = Device::gpu().unwrap();

        // Create input data (use same values for fair comparison)
        let x_data: Vec<f32> = (0..256).map(|i| i as f32).collect();

        // Create CPU conv with im2col algorithm
        let x_cpu = RawTensor::new(x_data.clone(), &[1, 4, 8, 8], true);
        let conv_cpu = Conv2d::new_on_device(4, 8, 3, 1, 0, true, Device::CPU);
        let weight_size = 4 * 8 * 3 * 3;
        conv_cpu.weight.borrow_mut().data = crate::storage::Storage::cpu(vec![1.0; weight_size]);
        conv_cpu.bias.as_ref().unwrap().borrow_mut().data =
            crate::storage::Storage::cpu(vec![0.0; 8]);
        conv_cpu.set_algo(ConvAlgo::Im2col);

        let y_cpu = conv_cpu.forward(&x_cpu);
        let loss_cpu = y_cpu.sum();
        loss_cpu.backward();

        // Create GPU conv with im2col algorithm
        let x_gpu = RawTensor::new(x_data, &[1, 4, 8, 8], true).to_device(device.clone());
        let conv_gpu = Conv2d::new_on_device(4, 8, 3, 1, 0, true, device.clone());
        conv_gpu.weight.borrow_mut().data = crate::storage::Storage::gpu(vec![1.0; weight_size]);
        conv_gpu.bias.as_ref().unwrap().borrow_mut().data =
            crate::storage::Storage::gpu(vec![0.0; 8]);
        conv_gpu.set_algo(ConvAlgo::Im2col);

        let y_gpu = conv_gpu.forward(&x_gpu);
        let loss_gpu = y_gpu.sum();
        loss_gpu.backward();

        // Compare input gradients (this tests col2im)
        let grad_cpu = x_cpu.grad().expect("CPU input should have gradient");
        let grad_gpu = x_gpu.grad().expect("GPU input should have gradient");

        let grad_cpu_data = grad_cpu.clone();
        let grad_gpu_data = grad_gpu.clone();

        assert_eq!(grad_cpu_data.len(), grad_gpu_data.len());

        for (i, (cpu_val, gpu_val)) in grad_cpu_data.iter().zip(grad_gpu_data.iter()).enumerate() {
            let abs_diff = (cpu_val - gpu_val).abs();
            assert!(
                abs_diff < 1e-4,
                "GPU col2im mismatch at index {i}: CPU={cpu_val}, GPU={gpu_val}, diff={abs_diff}"
            );
        }
    }

    #[test]
    fn test_conv2d_gpu_im2col_backward() {
        if Device::gpu().is_none() {
            return;
        }

        let device = Device::gpu().unwrap();

        // Create input on GPU with requires_grad=true
        let x_gpu = RawTensor::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            &[1, 1, 4, 4],
            true,
        )
        .to_device(device.clone());

        // Create conv with im2col algorithm
        let conv = Conv2d::new_on_device(1, 2, 3, 1, 0, true, device.clone());
        conv.set_algo(ConvAlgo::Im2col);

        // Forward pass
        let y = conv.forward(&x_gpu);

        // Backward pass (tests col2im)
        let loss = y.sum();
        loss.backward();

        // Verify we got an input gradient
        let grad_input = x_gpu.grad().expect("Input should have gradient");
        assert_eq!(grad_input.len(), 16);

        // Verify values are non-zero (col2im should have accumulated gradients)
        let grad_data = grad_input.clone();
        let has_non_zero = grad_data.iter().any(|&v| v.abs() > 1e-6);
        assert!(has_non_zero, "col2im gradient should have non-zero values");
    }

    #[test]
    fn test_conv2d_gpu_col2im_3x3_stride2() {
        if Device::gpu().is_none() {
            return;
        }

        let device = Device::gpu().unwrap();

        // Test with stride 2 (more challenging case)
        let x_data: Vec<f32> = (0..768).map(|i| i as f32).collect();

        // CPU version
        let x_cpu = RawTensor::new(x_data.clone(), &[1, 3, 16, 16], true);
        let conv_cpu = Conv2d::new_on_device(3, 8, 3, 2, 1, true, Device::CPU);
        let weight_size = 3 * 8 * 3 * 3;
        conv_cpu.weight.borrow_mut().data = crate::storage::Storage::cpu(vec![1.0; weight_size]);
        conv_cpu.bias.as_ref().unwrap().borrow_mut().data =
            crate::storage::Storage::cpu(vec![0.0; 8]);
        conv_cpu.set_algo(ConvAlgo::Im2col);

        let y_cpu = conv_cpu.forward(&x_cpu);
        let loss_cpu = y_cpu.sum();
        loss_cpu.backward();

        // GPU version
        let x_gpu = RawTensor::new(x_data, &[1, 3, 16, 16], true).to_device(device.clone());
        let conv_gpu = Conv2d::new_on_device(3, 8, 3, 2, 1, true, device.clone());
        conv_gpu.weight.borrow_mut().data = crate::storage::Storage::gpu(vec![1.0; weight_size]);
        conv_gpu.bias.as_ref().unwrap().borrow_mut().data =
            crate::storage::Storage::gpu(vec![0.0; 8]);
        conv_gpu.set_algo(ConvAlgo::Im2col);

        let y_gpu = conv_gpu.forward(&x_gpu);
        let loss_gpu = y_gpu.sum();
        loss_gpu.backward();

        // Compare gradients (col2im with stride 2)
        let grad_cpu = x_cpu.grad().expect("CPU input should have gradient");
        let grad_gpu = x_gpu.grad().expect("GPU input should have gradient");

        let grad_cpu_data = grad_cpu.clone();
        let grad_gpu_data = grad_gpu.clone();

        assert_eq!(grad_cpu_data.len(), grad_gpu_data.len());

        for (i, (cpu_val, gpu_val)) in grad_cpu_data.iter().zip(grad_gpu_data.iter()).enumerate() {
            let abs_diff = (cpu_val - gpu_val).abs();
            assert!(
                abs_diff < 1e-4,
                "GPU col2im stride2 mismatch at index {i}: CPU={cpu_val}, GPU={gpu_val}, diff={abs_diff}"
            );
        }
    }
}
