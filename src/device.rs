use std::fmt;
// ===== DEVICE ENUM =====

/// Compute device for tensor operations
///
/// In progress adding GPU support with wgpu.
#[derive(Debug, Clone, Default, PartialEq)]
pub enum Device {
    #[default]
    CPU,
    GPU(String),
}

impl Device {
    /// Check if this is a CPU device
    pub fn is_cpu(&self) -> bool {
        matches!(self, Device::CPU)
    }

    /// Check if this is a GPU device
    pub fn is_gpu(&self) -> bool {
        matches!(self, Device::GPU(_))
    }

    /// Get device name for display
    pub fn name(&self) -> &str {
        match self {
            Device::CPU => "CPU",
            Device::GPU(name) => name.as_str(),
        }
    }
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}
