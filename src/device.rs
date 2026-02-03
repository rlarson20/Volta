use std::fmt;
// ===== DEVICE ENUM =====

/// Compute device for tensor operations
///
/// In progress adding GPU support with wgpu.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum Device {
    #[default]
    CPU,
    GPU(String),
}

impl Device {
    /// Check if this is a CPU device
    #[must_use]
    pub const fn is_cpu(&self) -> bool {
        matches!(self, Self::CPU)
    }

    /// Check if this is a GPU device
    #[must_use]
    pub const fn is_gpu(&self) -> bool {
        matches!(self, Self::GPU(_))
    }

    /// Get device name for display
    #[must_use]
    pub const fn name(&self) -> &str {
        match self {
            Self::CPU => "CPU",
            Self::GPU(name) => name.as_str(),
        }
    }

    /// Get the default GPU device.
    ///
    /// Returns `Some(Device::GPU)` if GPU is available, `None` otherwise.
    /// The GPU name comes from the initialized GPU context.
    ///
    /// # Example
    /// ```no_run
    /// # #[cfg(feature = "gpu")]
    /// # {
    /// use volta::Device;
    ///
    /// if let Some(gpu) = Device::gpu() {
    ///     println!("Using GPU: {}", gpu.name());
    /// } else {
    ///     println!("GPU not available, using CPU");
    /// }
    /// # }
    /// ```
    #[cfg(feature = "gpu")]
    #[must_use]
    pub fn gpu() -> Option<Self> {
        crate::gpu::get_gpu_context().map(|ctx| Self::GPU(ctx.device_name().to_string()))
    }

    /// When GPU feature is disabled, gpu() always returns None
    #[cfg(not(feature = "gpu"))]
    pub fn gpu() -> Option<Self> {
        None
    }
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}
