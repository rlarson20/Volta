#[derive(Deserialize)]
pub struct ModelConfig {
    pub architecture: String,
    pub num_layers: usize,
    pub hidden_size: usize,
    // ... architecture-specific fields
}

pub fn load_config(path: &str) -> Result<ModelConfig>;
pub fn build_from_config(config: &ModelConfig) -> Result<Box<dyn Module>>;

//3. **Implement builders** for common architectures:
// - MLP builder
// - CNN builder
// - Simple Transformer builder (if layers exist)
//4. `src/lib.rs`, export config utilities
//
// **Tests**:
// - Parse simple MLP config
// - Build model from config
// - Validate architecture matches config
