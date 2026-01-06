#!/usr/bin/env python3
"""
Export a simple PyTorch MLP to SafeTensors format for loading in Volta.

This script demonstrates the external model loading workflow:
1. Define a simple PyTorch model (MLP for MNIST)
2. Export weights to SafeTensors format
3. Generate test data for validation

Requirements:
    pip install torch safetensors

Usage:
    python scripts/export_pytorch_mnist.py
"""

import os
import json
import torch
import torch.nn as nn

try:
    from safetensors.torch import save_file
except ImportError:
    print("Error: safetensors package not found")
    print("Please install it with: pip install safetensors")
    exit(1)


class SimpleMLP(nn.Module):
    """Simple MLP for MNIST digit classification.

    Architecture:
        Input (784) → Linear (128) → ReLU → Linear (10) → Output

    This architecture matches the Volta example in load_external_mnist.rs
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128, bias=True)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10, bias=True)

    def forward(self, x):
        """Forward pass through the network."""
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def print_weight_info(model):
    """Print information about model weights for debugging."""
    print("\n=== Model Weight Information ===")
    for name, param in model.state_dict().items():
        print(f"{name:20s} shape: {list(param.shape):20s} dtype: {param.dtype}")
    print()


def export_model():
    """Export PyTorch model to SafeTensors format."""
    print("=== PyTorch Model Export ===\n")

    # Create model
    print("Creating SimpleMLP model...")
    model = SimpleMLP()

    # Set model to evaluation mode (disables dropout, etc.)
    # Note: This is PyTorch's .eval() method, not Python's eval() function
    model.train(False)

    # Print weight information
    print_weight_info(model)

    # Create output directory if it doesn't exist
    os.makedirs("models", exist_ok=True)

    # Save weights to SafeTensors
    output_path = "models/pytorch_mnist.safetensors"
    print(f"Saving weights to {output_path}...")
    state_dict = model.state_dict()

    # Convert to contiguous tensors (required by safetensors)
    state_dict = {k: v.contiguous() for k, v in state_dict.items()}
    save_file(state_dict, output_path)
    print(f"✓ Weights saved successfully")

    # Generate test data for validation
    print("\nGenerating test data...")
    num_test_samples = 5
    torch.manual_seed(42)  # For reproducibility
    test_inputs = torch.randn(num_test_samples, 784)

    with torch.no_grad():
        test_outputs = model(test_inputs)

    # Save test data as JSON
    test_data_path = "models/pytorch_mnist_test.json"
    test_data = {
        "inputs": test_inputs.tolist(),
        "outputs": test_outputs.tolist(),
        "num_samples": num_test_samples,
        "input_shape": [784],
        "output_shape": [10]
    }

    with open(test_data_path, "w") as f:
        json.dump(test_data, f, indent=2)

    print(f"✓ Test data saved to {test_data_path}")
    print(f"  - {num_test_samples} test samples")
    print(f"  - Input shape: {test_data['input_shape']}")
    print(f"  - Output shape: {test_data['output_shape']}")

    # Print summary
    print("\n=== Export Summary ===")
    print(f"Model architecture: Linear(784→128) → ReLU → Linear(128→10)")
    print(f"Weights file: {output_path}")
    print(f"Test data file: {test_data_path}")
    print("\nNote: PyTorch Linear weights are stored as [out_features, in_features]")
    print("      Volta expects [in_features, out_features], so transposition is needed")
    print("\nNext steps:")
    print("  1. Run the Rust example: cargo run --example load_external_mnist")
    print("  2. The example will load these weights and validate outputs")


if __name__ == "__main__":
    export_model()
