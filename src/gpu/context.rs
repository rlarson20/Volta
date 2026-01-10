//! GPU context management
//!
//! The GpuContext holds the wgpu device and queue, which are needed
//! for all GPU operations. Think of it as your "connection" to the GPU.

/// Manages the GPU device, queue, and compiled compute pipelines
pub struct GpuContext {
    /// The GPU device - represents the actual hardware
    device: wgpu::Device,
    /// Command queue - where we submit work to the GPU
    queue: wgpu::Queue,
    /// Adapter info for debugging
    adapter_info: wgpu::AdapterInfo,
    /// Pre-compiled compute pipelines for common operations
    pipelines: ComputePipelines,
}

/// Collection of pre-compiled compute pipelines
///
/// Compiling shaders is expensive, so we do it once at initialization
/// and reuse the pipelines for all operations.
pub struct ComputePipelines {
    // Element-wise operations
    pub add: wgpu::ComputePipeline,
    pub sub: wgpu::ComputePipeline,
    pub mul: wgpu::ComputePipeline,
    pub div: wgpu::ComputePipeline,
    pub max: wgpu::ComputePipeline,
    pub mod_op: wgpu::ComputePipeline,
    pub cmplt: wgpu::ComputePipeline,

    // Unary operations
    pub neg: wgpu::ComputePipeline,
    pub exp: wgpu::ComputePipeline,
    pub log: wgpu::ComputePipeline,
    pub relu: wgpu::ComputePipeline,
    pub sigmoid: wgpu::ComputePipeline,
    pub tanh: wgpu::ComputePipeline,
    pub sqrt: wgpu::ComputePipeline,
    pub recip: wgpu::ComputePipeline,
    pub exp2: wgpu::ComputePipeline,
    pub log2: wgpu::ComputePipeline,
    pub sin: wgpu::ComputePipeline,
    pub cos: wgpu::ComputePipeline,

    // Unary backward operations
    pub neg_backward: wgpu::ComputePipeline,
    pub exp_backward: wgpu::ComputePipeline,
    pub log_backward: wgpu::ComputePipeline,
    pub relu_backward: wgpu::ComputePipeline,
    pub sigmoid_backward: wgpu::ComputePipeline,
    pub tanh_backward: wgpu::ComputePipeline,
    pub sqrt_backward: wgpu::ComputePipeline,
    pub recip_backward: wgpu::ComputePipeline,
    pub exp2_backward: wgpu::ComputePipeline,
    pub log2_backward: wgpu::ComputePipeline,
    pub sin_backward: wgpu::ComputePipeline,
    pub cos_backward: wgpu::ComputePipeline,

    // Binary backward operations
    pub add_backward_a: wgpu::ComputePipeline,
    pub add_backward_b: wgpu::ComputePipeline,
    pub sub_backward_a: wgpu::ComputePipeline,
    pub sub_backward_b: wgpu::ComputePipeline,
    pub mul_backward_a: wgpu::ComputePipeline,
    pub mul_backward_b: wgpu::ComputePipeline,
    pub div_backward_a: wgpu::ComputePipeline,
    pub div_backward_b: wgpu::ComputePipeline,
    pub max_backward_a: wgpu::ComputePipeline,
    pub max_backward_b: wgpu::ComputePipeline,

    // Reductions
    pub sum_reduce: wgpu::ComputePipeline,
    pub max_reduce: wgpu::ComputePipeline,
    pub mean_reduce: wgpu::ComputePipeline,

    // Movement operations
    pub permute: wgpu::ComputePipeline,
    pub expand: wgpu::ComputePipeline,
    pub pad: wgpu::ComputePipeline,
    pub shrink: wgpu::ComputePipeline,
    pub stride: wgpu::ComputePipeline,

    // Matrix multiplication (this is the big one for ML!)
    pub matmul: wgpu::ComputePipeline,

    // Matrix multiplication backward
    pub matmul_backward_a: wgpu::ComputePipeline,
    pub matmul_backward_b: wgpu::ComputePipeline,

    // Reduction backward
    pub sum_backward: wgpu::ComputePipeline,
    pub mean_backward: wgpu::ComputePipeline,
    pub max_backward: wgpu::ComputePipeline,

    // Optimizer step
    pub optimizer_step: wgpu::ComputePipeline,

    // Image-to-column transformation for convolution
    pub im2col: wgpu::ComputePipeline,
}

impl GpuContext {
    /// Initialize the GPU context
    ///
    /// This is an expensive operation that:
    /// 1. Finds a suitable GPU adapter
    /// 2. Creates a device and queue
    /// 3. Compiles all our compute shaders
    pub fn new() -> Result<Self, String> {
        // wgpu is async, but we want a sync API for simplicity
        // pollster::block_on runs async code synchronously
        pollster::block_on(Self::new_async())
    }

    async fn new_async() -> Result<Self, String> {
        // Step 1: Create a wgpu instance
        // This is the entry point to wgpu
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(), // Try all available backends (Vulkan, Metal, DX12, etc.)
            ..Default::default()
        });

        // Step 2: Request an adapter (represents a physical GPU)
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None, // We don't need a surface for compute
                force_fallback_adapter: false,
            })
            .await
            .map_err(|e| format!("No suitable GPU adapter found: {e}"))?;

        let adapter_info = adapter.get_info();

        // Step 3: Request a device (logical connection to the GPU)

        let device_descriptor = wgpu::DeviceDescriptor {
            label: Some("Volta GPU Device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::Performance,
            ..Default::default()
        };

        let (device, queue) = adapter
            .request_device(&device_descriptor)
            .await
            .map_err(|e| format!("Failed to create device: {}", e))?;
        // Step 4: Compile all our compute shaders
        let pipelines = Self::create_pipelines(&device)?;

        Ok(GpuContext {
            device,
            queue,
            adapter_info,
            pipelines,
        })
    }

    /// Get the GPU device name for display
    pub fn device_name(&self) -> &str {
        &self.adapter_info.name
    }

    /// Get a reference to the wgpu device
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// Get a reference to the command queue
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// Get the compiled pipelines
    pub fn pipelines(&self) -> &ComputePipelines {
        &self.pipelines
    }

    /// Create all compute pipelines by compiling shaders
    fn create_pipelines(device: &wgpu::Device) -> Result<ComputePipelines, String> {
        // Load and compile shader modules
        let elementwise_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Elementwise Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/elementwise.wgsl").into()),
        });

        let unary_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Unary Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/unary.wgsl").into()),
        });

        let reduce_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Reduce Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/reduce.wgsl").into()),
        });

        let matmul_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("MatMul Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/matmul.wgsl").into()),
        });

        let movement_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Movement Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/movement.wgsl").into()),
        });

        let unary_backward_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Unary Backward Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/unary_backward.wgsl").into()),
        });

        let binary_backward_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Binary Backward Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/binary_backward.wgsl").into()),
        });

        let matmul_backward_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("MatMul Backward Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/matmul_backward.wgsl").into()),
        });

        let reduce_backward_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Reduce Backward Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/reduce_backward.wgsl").into()),
        });

        let optimizer_step_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Optimizer Step Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/optimizer_step.wgsl").into()),
        });

        let im2col_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Im2col Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/im2col.wgsl").into()),
        });

        // Helper to create a compute pipeline
        let create_pipeline = |shader: &wgpu::ShaderModule, entry_point: &str, label: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: None, // Auto-generate layout from shader
                module: shader,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        Ok(ComputePipelines {
            // Element-wise binary ops
            add: create_pipeline(&elementwise_shader, "add", "Add Pipeline"),
            sub: create_pipeline(&elementwise_shader, "sub", "Sub Pipeline"),
            mul: create_pipeline(&elementwise_shader, "mul", "Mul Pipeline"),
            div: create_pipeline(&elementwise_shader, "div", "Div Pipeline"),
            max: create_pipeline(&elementwise_shader, "max_elem", "Max Pipeline"),
            mod_op: create_pipeline(&elementwise_shader, "mod_op", "Mod Pipeline"),
            cmplt: create_pipeline(&elementwise_shader, "cmplt", "Cmplt Pipeline"),

            // Unary ops
            neg: create_pipeline(&unary_shader, "neg", "Neg Pipeline"),
            exp: create_pipeline(&unary_shader, "exp_op", "Exp Pipeline"),
            log: create_pipeline(&unary_shader, "log_op", "Log Pipeline"),
            relu: create_pipeline(&unary_shader, "relu", "ReLU Pipeline"),
            sigmoid: create_pipeline(&unary_shader, "sigmoid", "Sigmoid Pipeline"),
            tanh: create_pipeline(&unary_shader, "tanh_op", "Tanh Pipeline"),
            sqrt: create_pipeline(&unary_shader, "sqrt_op", "Sqrt Pipeline"),
            recip: create_pipeline(&unary_shader, "recip", "Recip Pipeline"),
            exp2: create_pipeline(&unary_shader, "exp2_op", "Exp2 Pipeline"),
            log2: create_pipeline(&unary_shader, "log2_op", "Log2 Pipeline"),
            sin: create_pipeline(&unary_shader, "sin_op", "Sin Pipeline"),
            cos: create_pipeline(&unary_shader, "cos_op", "Cos Pipeline"),

            // Unary backward ops
            neg_backward: create_pipeline(
                &unary_backward_shader,
                "neg_backward",
                "Neg Backward Pipeline",
            ),
            exp_backward: create_pipeline(
                &unary_backward_shader,
                "exp_backward",
                "Exp Backward Pipeline",
            ),
            log_backward: create_pipeline(
                &unary_backward_shader,
                "log_backward",
                "Log Backward Pipeline",
            ),
            relu_backward: create_pipeline(
                &unary_backward_shader,
                "relu_backward",
                "ReLU Backward Pipeline",
            ),
            sigmoid_backward: create_pipeline(
                &unary_backward_shader,
                "sigmoid_backward",
                "Sigmoid Backward Pipeline",
            ),
            tanh_backward: create_pipeline(
                &unary_backward_shader,
                "tanh_backward",
                "Tanh Backward Pipeline",
            ),
            sqrt_backward: create_pipeline(
                &unary_backward_shader,
                "sqrt_backward",
                "Sqrt Backward Pipeline",
            ),
            recip_backward: create_pipeline(
                &unary_backward_shader,
                "recip_backward",
                "Recip Backward Pipeline",
            ),
            exp2_backward: create_pipeline(
                &unary_backward_shader,
                "exp2_backward",
                "Exp2 Backward Pipeline",
            ),
            log2_backward: create_pipeline(
                &unary_backward_shader,
                "log2_backward",
                "Log2 Backward Pipeline",
            ),
            sin_backward: create_pipeline(
                &unary_backward_shader,
                "sin_backward",
                "Sin Backward Pipeline",
            ),
            cos_backward: create_pipeline(
                &unary_backward_shader,
                "cos_backward",
                "Cos Backward Pipeline",
            ),

            // Binary backward ops
            add_backward_a: create_pipeline(
                &binary_backward_shader,
                "add_backward_a",
                "Add Backward A Pipeline",
            ),
            add_backward_b: create_pipeline(
                &binary_backward_shader,
                "add_backward_b",
                "Add Backward B Pipeline",
            ),
            sub_backward_a: create_pipeline(
                &binary_backward_shader,
                "sub_backward_a",
                "Sub Backward A Pipeline",
            ),
            sub_backward_b: create_pipeline(
                &binary_backward_shader,
                "sub_backward_b",
                "Sub Backward B Pipeline",
            ),
            mul_backward_a: create_pipeline(
                &binary_backward_shader,
                "mul_backward_a",
                "Mul Backward A Pipeline",
            ),
            mul_backward_b: create_pipeline(
                &binary_backward_shader,
                "mul_backward_b",
                "Mul Backward B Pipeline",
            ),
            div_backward_a: create_pipeline(
                &binary_backward_shader,
                "div_backward_a",
                "Div Backward A Pipeline",
            ),
            div_backward_b: create_pipeline(
                &binary_backward_shader,
                "div_backward_b",
                "Div Backward B Pipeline",
            ),
            max_backward_a: create_pipeline(
                &binary_backward_shader,
                "max_backward_a",
                "Max Backward A Pipeline",
            ),
            max_backward_b: create_pipeline(
                &binary_backward_shader,
                "max_backward_b",
                "Max Backward B Pipeline",
            ),

            // Reductions
            sum_reduce: create_pipeline(&reduce_shader, "sum_reduce", "Sum Reduce Pipeline"),
            max_reduce: create_pipeline(&reduce_shader, "max_reduce", "Max Reduce Pipeline"),
            mean_reduce: create_pipeline(&reduce_shader, "mean_reduce", "Mean Reduce Pipeline"),

            // Movement operations
            permute: create_pipeline(&movement_shader, "permute", "Permute Pipeline"),
            expand: create_pipeline(&movement_shader, "expand", "Expand Pipeline"),
            pad: create_pipeline(&movement_shader, "pad", "Pad Pipeline"),
            shrink: create_pipeline(&movement_shader, "shrink", "Shrink Pipeline"),
            stride: create_pipeline(&movement_shader, "stride", "Stride Pipeline"),

            // Matrix multiplication
            matmul: create_pipeline(&matmul_shader, "matmul", "MatMul Pipeline"),

            // Matrix multiplication backward
            matmul_backward_a: create_pipeline(
                &matmul_backward_shader,
                "matmul_backward_a",
                "MatMul Backward A Pipeline",
            ),
            matmul_backward_b: create_pipeline(
                &matmul_backward_shader,
                "matmul_backward_b",
                "MatMul Backward B Pipeline",
            ),

            // Reduction backward
            sum_backward: create_pipeline(&reduce_backward_shader, "main", "Sum Backward Pipeline"),
            mean_backward: create_pipeline(
                &reduce_backward_shader,
                "main",
                "Mean Backward Pipeline",
            ),
            max_backward: create_pipeline(&reduce_backward_shader, "main", "Max Backward Pipeline"),

            // Optimizer step
            optimizer_step: create_pipeline(
                &optimizer_step_shader,
                "main",
                "Optimizer Step Pipeline",
            ),

            // Image-to-column transformation
            im2col: create_pipeline(&im2col_shader, "im2col_main", "Im2col Pipeline"),
        })
    }
}
