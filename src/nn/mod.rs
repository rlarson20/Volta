use crate::device::Device;
use crate::dtype::DType;
use crate::io::StateDict;
use crate::tensor::Tensor;

/// Hyperparameter checks for the layers' `try_new` (#133).
pub(crate) mod hyper;
/// Layers and containers.
pub(crate) mod layers;
/// Atomic state-dict loading (#65).
mod load_plan;
/// Optimizers.
pub(crate) mod optim;
pub(crate) mod transformer;

pub use layers::{
    BatchNorm1d, BatchNorm2d, Conv2d, ConvAlgo, ConvTranspose2d, Dropout, Embedding, Flatten, GELU,
    LSTMCell, LayerNorm, Linear, MaxPool2d, PixelShuffle, ReLU, Sequential, SequentialBuilder,
    Sigmoid, Tanh,
};
pub use load_plan::LoadPlan;
pub use optim::{Adam, Muon, Optimizer, SGD};
pub use transformer::{
    FeedForward, MultiHeadAttention, PositionalEncoding, PositionalEncodingType,
    ScaledDotProductAttention, TransformerBlock, causal_mask,
};

/// PyTorch's mixed-dtype rule for the norm layers (audit A5, #316): a half (f16/bf16)
/// input whose parameters are f32 or the input's own dtype is normalized in f32 and
/// returned in the input's dtype. Under `amp::autocast` the activations stay low and the
/// f32 parameters keep f32 gradients; with half parameters (`Module::to_dtype`) the
/// statistics are still taken in f32, as PyTorch's half norm kernels accumulate in f32,
/// so a variance whose sum passes f16's 65504 does not overflow.
///
/// `f` gets the input in the dtype to compute in, and that dtype: the layer casts its
/// parameters (and running statistics) to it. The casts are explicit graph nodes, so
/// `DTypePolicy::Strict` accepts them and half parameters get half gradients. Any other
/// pair goes to `f` with the parameters' dtype, which promotes or refuses as usual.
pub(crate) fn norm_mixed(
    x: &Tensor,
    param: crate::DType,
    f: impl FnOnce(&Tensor, crate::DType) -> Tensor,
) -> Tensor {
    use crate::DType;
    let d = x.dtype();
    if matches!(d, DType::F16 | DType::BF16) && (param == DType::F32 || param == d) {
        f(&x.to_dtype(DType::F32), DType::F32).to_dtype(d)
    } else {
        f(x, param)
    }
}

/// `params` with each tensor once, first occurrence kept, by identity ([`Tensor::ptr_eq`]).
/// A container's `parameters()` goes through this so a weight tied between two layers is
/// listed (and so stepped, saved and zeroed) once, as in PyTorch's `Module.parameters()` (#101).
pub(crate) fn unique_parameters(params: impl IntoIterator<Item = Tensor>) -> Vec<Tensor> {
    let mut seen = std::collections::HashSet::new();
    params
        .into_iter()
        .filter(|p| seen.insert(p.as_ptr()))
        .collect()
}

/// Adds each entry of `child` (a sub-module's `state_dict()`) to `state` under `prefix`,
/// which ends in `.` (`"ffn."` turns `expand.weight` into `ffn.expand.weight`).
pub(crate) fn insert_prefixed(state: &mut StateDict, prefix: &str, child: StateDict) {
    state.extend(
        child
            .into_iter()
            .map(|(key, value)| (format!("{prefix}{key}"), value)),
    );
}

/// Private supertrait that keeps [`RecurrentLayer`] implementable only inside
/// this crate. `Module` and [`Layer`] stay open so users can write custom layers.
mod sealed {
    /// Marker for the sealed layer traits.
    pub trait SealedLayer {}
}

pub(crate) use sealed::SealedLayer;

/// Anything that owns learnable parameters and can be saved, loaded, and moved between devices.
///
/// This is the base of the layer traits. It has no `forward`, because the input contract differs
/// by kind of layer: see [`Layer`] (one tensor in, one out; [`Embedding`] takes an integer index
/// tensor) and [`RecurrentLayer`] (input plus carried state).
///
/// Open for implementation outside the crate: a method added to this trait in a later
/// version comes with a default body, so existing implementations keep compiling.
pub trait Module {
    /// Every learnable tensor of the module, in a stable order. A tensor shared between
    /// layers (a tied weight) is listed once.
    fn parameters(&self) -> Vec<Tensor>;

    // State dict methods
    /// The parameters as a name-to-values map, ready to save with [`crate::io`]. Each
    /// entry keeps its parameter's dtype.
    fn state_dict(&self) -> StateDict;

    /// Names, in `plan`, every tensor this module loads from a state dict: each of its
    /// own with [`LoadPlan::param`] (under the same name as in [`Self::state_dict`]) and
    /// each sub-module with [`LoadPlan::child`] (under its prefix). This is the method a
    /// layer implements; [`Self::load_state_dict`] and [`Self::load_state_dict_partial`]
    /// build a plan with it, check every write and only then make them (#65).
    ///
    /// The names must be exactly the keys of [`Self::state_dict`], or a strict load
    /// expects keys the module never saves.
    ///
    /// ```
    /// use volta::Tensor;
    /// use volta::io::{StateDict, TensorData};
    /// use volta::nn::{LoadPlan, Module};
    ///
    /// /// A learnable per-feature scale.
    /// struct Scale {
    ///     /// The scale, shape `[n]`.
    ///     weight: Tensor,
    /// }
    ///
    /// impl Module for Scale {
    ///     fn parameters(&self) -> Vec<Tensor> {
    ///         vec![self.weight.clone()]
    ///     }
    ///     fn state_dict(&self) -> StateDict {
    ///         [("weight".to_string(), TensorData::from_tensor(&self.weight))].into()
    ///     }
    ///     fn plan_load<'a>(&self, plan: &mut LoadPlan<'a>) {
    ///         plan.param("weight", &self.weight);
    ///     }
    /// }
    /// ```
    fn plan_load<'a>(&self, plan: &mut LoadPlan<'a>);

    /// Loads `state` into this module, strictly, as PyTorch's default `strict=True`.
    ///
    /// The keys of `state` must be exactly the keys the module loads (those of
    /// [`Self::state_dict`]): a missing or unexpected key is an `Err` naming every such
    /// key. A typo in a key prefix therefore fails instead of leaving the module at its
    /// random initialization (#344). Each entry then loads as in
    /// [`Self::load_state_dict_partial`]: it keeps the parameter's shape, device, and
    /// `requires_grad`, and an entry in another dtype is cast to the parameter's.
    ///
    /// The load is atomic: every entry is checked before any is written, so on `Err` the
    /// module is unchanged (#65). No weight is copied to check the keys.
    ///
    /// # Errors
    ///
    /// `VoltaError::StateKeyMismatch` if a key is missing or unexpected, and otherwise
    /// the errors of [`Self::load_state_dict_partial`]. Nothing is written on `Err`.
    ///
    /// # Panics
    ///
    /// If writing a checked entry fails (a GPU upload), leaving the module partly loaded.
    fn load_state_dict(&mut self, state: &StateDict) -> crate::Result<()> {
        let plan = LoadPlan::of(&*self, state);
        plan.check_keys()?;
        plan.apply()
    }

    /// Loads the entries of `state` that match this module's parameters, leniently.
    ///
    /// Keys the module does not have are ignored, and parameters with no entry keep
    /// their values (use [`crate::io::load_state_dict_checked`] to see which). An entry
    /// that is present must fit: it keeps the parameter's shape, device, and
    /// `requires_grad`, and only the values change. An entry in another dtype is cast to
    /// the parameter's (following [`crate::storage::Storage::cast`]).
    ///
    /// The load is atomic: every matching entry is checked before any is written, so on
    /// `Err` the module is unchanged (#65).
    ///
    /// # Errors
    ///
    /// `VoltaError::StateShapeMismatch` if an entry's shape differs from its
    /// parameter's, `VoltaError::StateDataLength` if an entry has the wrong number of
    /// bytes for its own shape and dtype (`InvalidParameter` for a bool byte that is not
    /// 0 or 1), and `VoltaError::DTypeMismatch` if the parameter is not a float tensor.
    /// The `key` is the entry's name in `state`. A checkpoint is data from outside the
    /// program, so this is an `Err`, not a panic.
    ///
    /// # Panics
    ///
    /// As [`Self::load_state_dict`].
    fn load_state_dict_partial(&mut self, state: &StateDict) -> crate::Result<()> {
        LoadPlan::of(&*self, state).apply()
    }

    /// Clears the gradient of every parameter.
    fn zero_grad(&mut self) {
        for p in self.parameters() {
            p.zero_grad();
        }
    }

    /// Move all module parameters to a specific device
    ///
    /// This method transfers all parameters (weights, biases, etc.) to the specified device.
    /// The default implementation moves each parameter individually by updating their
    /// storage and device in place.
    ///
    /// # Arguments
    /// * `device` - Target device (CPU or GPU)
    ///
    /// # Example
    /// ```no_run
    /// # use volta::nn::{Linear, Module};
    /// # use volta::Device;
    /// # #[cfg(feature = "gpu")]
    /// # {
    /// let mut layer = Linear::new(784, 128, true);
    /// let device = Device::gpu().expect("GPU required");
    /// layer.to_device(device);
    /// // All parameters are now on GPU
    /// # }
    /// ```
    fn to_device(&mut self, device: Device) {
        for param in self.parameters() {
            // Moves the data and any gradient. `Storage::to_device` panics when the GPU is
            // unavailable rather than leaving the data on the CPU under a GPU label.
            param.move_to_device(&device);
        }
    }

    /// Converts every parameter, and its gradient, to `dtype` in place: PyTorch's
    /// `module.to(dtype)` (`.double()`, `.half()`). Values follow [`crate::storage::Storage::cast`],
    /// so f32 to f64 is exact. A layer holding a non-parameter tensor (`BatchNorm`'s running
    /// statistics, `PositionalEncoding`'s table) overrides this to convert it too, and a
    /// container (`Sequential`) forwards it to each child.
    ///
    /// Of the layers, those whose kernels are dtype-generic run in f64 after this; see
    /// ADDENDUM §11 for which.
    ///
    /// # Panics
    /// If `dtype` is not a float dtype (parameters need gradients), or a parameter is on
    /// the GPU and `dtype` is F64 (GPU tensors hold F32, F16, BF16 and Bool).
    fn to_dtype(&mut self, dtype: DType) {
        cast_parameters(self, dtype);
    }

    /// Switch between training and evaluation modes.
    /// Important for layers like `BatchNorm` and Dropout.
    fn train(&mut self, _mode: bool) {}
    /// Switches to evaluation mode; same as `train(false)`.
    fn eval(&mut self) {
        self.train(false);
    }
}

/// The default of [`Module::to_dtype`]: casts every parameter of `module` to `dtype`.
/// Overrides call it and then convert their buffers.
///
/// # Panics
/// As [`Module::to_dtype`].
#[track_caller]
pub(crate) fn cast_parameters<M: Module + ?Sized>(module: &M, dtype: DType) {
    assert!(
        dtype.is_float(),
        "to_dtype({dtype}): a module's parameters need a float dtype"
    );
    for param in module.parameters() {
        param.cast_in_place(dtype);
    }
}

/// A layer that maps one tensor to one tensor. Only `Layer`s can go in a [`Sequential`].
///
/// Open for implementation outside the crate: a method added to this trait in a later
/// version comes with a default body, so existing implementations keep compiling.
pub trait Layer: Module {
    /// Run the layer on `x`.
    fn forward(&self, x: &Tensor) -> Tensor;
}

/// A layer that carries state between steps, such as [`LSTMCell`]. Sealed: only this crate
/// implements it.
pub trait RecurrentLayer: Module + SealedLayer {
    /// State carried from one step to the next.
    type State;
    /// Zero state for a batch of `batch` sequences.
    fn initial_state(&self, batch: usize) -> Self::State;
    /// Advance one step: update `state` in place and return the step output.
    fn step(&self, x: &Tensor, state: &mut Self::State) -> Tensor;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::TensorData;

    #[test]
    fn insert_prefixed_prepends_the_prefix_to_every_key() {
        let entry = || TensorData::from_f32(vec![1.0], vec![1]);
        let child: StateDict = [("w".to_string(), entry()), ("x.y".to_string(), entry())].into();
        let mut state = StateDict::new();
        insert_prefixed(&mut state, "p.", child);
        let keys: Vec<&str> = state.keys().map(String::as_str).collect();
        assert_eq!(keys, ["p.w", "p.x.y"]);
    }
}
