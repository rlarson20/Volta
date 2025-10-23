// ===== DEVICE ENUM =====

/// Compute device for tensor operations
///
/// Currently only CPU is implemented. GPU would require integration
/// with CUDA/OpenCL, and Metal would be for Apple Silicon.
#[derive(Debug, Clone, Default)]
pub enum Device {
    #[default]
    CPU,
    GPU,
    //TODO: possible Metal variant
}
