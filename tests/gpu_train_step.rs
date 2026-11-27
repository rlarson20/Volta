//! The GPU gate for #464: every built-in layer runs a full training step on the GPU (forward,
//! a randomly weighted loss, backward, one `Adam` step) and matches the same step on the CPU.
//!
//! Each case's GPU step runs under `DevicePolicy::Strict`, so any op that would compute on
//! the CPU for GPU inputs (a device-policy gap) panics. A panicking case is caught and
//! reported with the others, so one run shows every failing layer. `gpu_counters` checks
//! that the step never waits on the host (no readback, no queue drain).
//!
//! Skips cleanly when no GPU adapter is available.

#![cfg(feature = "gpu")]

#[path = "common/mod.rs"]
mod common;

use std::panic::{AssertUnwindSafe, catch_unwind};

use common::{TOL_MATMUL_BWD, gpu_device, gpu_leaf, skip_if_no_gpu};
use volta::io::StateDict;
use volta::nn::{
    Adam, BatchNorm1d, BatchNorm2d, Conv2d, ConvTranspose2d, Dropout, Embedding, FeedForward,
    Flatten, GELU, LSTMCell, Layer, LayerNorm, Linear, LoadPlan, MaxPool2d, Module,
    MultiHeadAttention, PixelShuffle, PositionalEncoding, PositionalEncodingType, ReLU,
    RecurrentLayer, Sequential, Sigmoid, Tanh, TransformerBlock,
};
use volta::{DevicePolicy, RawTensor, Tensor, manual_seed, with_device_policy};

/// One built-in layer, built fresh on the CPU each time it is called.
type Build = fn() -> Box<dyn Layer>;

/// What a case feeds the layer.
#[derive(Clone, Copy)]
enum Input {
    /// A random f32 leaf of this shape that requires grad.
    Float(&'static [usize]),
    /// Fixed I64 class indices (for `Embedding`), which carry no gradient.
    Indices,
}

/// How a case's numbers are compared with the CPU.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Check {
    /// Output, input gradient, parameter gradients and post-step parameters.
    Full,
    /// Only the gap set and that everything stays on the GPU: the layer draws random
    /// numbers whose stream differs between devices (`Dropout`).
    DeviceOnly,
}

/// A layer and its input.
struct Case {
    /// The name in failure messages.
    name: &'static str,
    /// Builds the layer on the CPU.
    build: Build,
    /// The input.
    input: Input,
    /// How results are compared.
    check: Check,
}

/// `LSTMCell` unrolled two steps from the zero state, as a `Layer`.
struct TwoSteps(LSTMCell);

impl Module for TwoSteps {
    fn parameters(&self) -> Vec<Tensor> {
        self.0.parameters()
    }
    fn state_dict(&self) -> StateDict {
        self.0.state_dict()
    }
    fn plan_load<'a>(&self, plan: &mut LoadPlan<'a>) {
        self.0.plan_load(plan);
    }
    fn to_device(&mut self, device: volta::Device) {
        self.0.to_device(device);
    }
}

impl Layer for TwoSteps {
    fn forward(&self, x: &Tensor) -> Tensor {
        let batch = x
            .shape()
            .dim(volta::Axis(0))
            .expect("input has a batch axis");
        let mut state = self.0.initial_state(batch);
        let _ = self.0.step(x, &mut state);
        self.0.step(x, &mut state)
    }
}

/// `MultiHeadAttention` with a causal mask, as a `Layer`.
struct CausalMha(MultiHeadAttention);

impl Module for CausalMha {
    fn parameters(&self) -> Vec<Tensor> {
        self.0.parameters()
    }
    fn state_dict(&self) -> StateDict {
        self.0.state_dict()
    }
    fn plan_load<'a>(&self, plan: &mut LoadPlan<'a>) {
        self.0.plan_load(plan);
    }
    fn to_device(&mut self, device: volta::Device) {
        self.0.to_device(device);
    }
}

impl Layer for CausalMha {
    fn forward(&self, x: &Tensor) -> Tensor {
        self.0.forward(x, true, None)
    }
}

/// The cases, one per built-in layer (and a two-layer MLP).
const CASES: &[Case] = &[
    Case {
        name: "Linear (bias)",
        build: || Box::new(Linear::new(16, 8, true)),
        input: Input::Float(&[4, 16]),
        check: Check::Full,
    },
    Case {
        name: "Linear (no bias)",
        build: || Box::new(Linear::new(16, 8, false)),
        input: Input::Float(&[4, 16]),
        check: Check::Full,
    },
    Case {
        name: "MLP",
        build: || {
            Box::new(Sequential::new(vec![
                Box::new(Linear::new(16, 32, true)),
                Box::new(ReLU),
                Box::new(Linear::new(32, 8, true)),
            ]))
        },
        input: Input::Float(&[4, 16]),
        check: Check::Full,
    },
    Case {
        name: "Conv2d",
        build: || Box::new(Conv2d::new(3, 4, 3, 1, 1, true)),
        input: Input::Float(&[2, 3, 8, 8]),
        check: Check::Full,
    },
    Case {
        name: "ConvTranspose2d",
        build: || Box::new(ConvTranspose2d::new(3, 4, 3, 2, 1, true)),
        input: Input::Float(&[2, 3, 5, 5]),
        check: Check::Full,
    },
    Case {
        name: "MaxPool2d",
        build: || Box::new(MaxPool2d::new(2, 2, 0)),
        input: Input::Float(&[2, 3, 8, 8]),
        check: Check::Full,
    },
    Case {
        name: "BatchNorm1d",
        build: || Box::new(BatchNorm1d::new(16)),
        input: Input::Float(&[4, 16]),
        check: Check::Full,
    },
    Case {
        name: "BatchNorm2d",
        build: || Box::new(BatchNorm2d::new(3)),
        input: Input::Float(&[2, 3, 6, 6]),
        check: Check::Full,
    },
    Case {
        name: "LayerNorm",
        build: || Box::new(LayerNorm::new(vec![16])),
        input: Input::Float(&[4, 16]),
        check: Check::Full,
    },
    Case {
        name: "GELU",
        build: || Box::new(GELU),
        input: Input::Float(&[4, 16]),
        check: Check::Full,
    },
    Case {
        name: "ReLU",
        build: || Box::new(ReLU),
        input: Input::Float(&[4, 16]),
        check: Check::Full,
    },
    Case {
        name: "Sigmoid",
        build: || Box::new(Sigmoid),
        input: Input::Float(&[4, 16]),
        check: Check::Full,
    },
    Case {
        name: "Tanh",
        build: || Box::new(Tanh),
        input: Input::Float(&[4, 16]),
        check: Check::Full,
    },
    Case {
        name: "Flatten",
        build: || Box::new(Flatten),
        input: Input::Float(&[2, 3, 4, 4]),
        check: Check::Full,
    },
    Case {
        name: "PixelShuffle",
        build: || Box::new(PixelShuffle::new(2)),
        input: Input::Float(&[2, 8, 3, 3]),
        check: Check::Full,
    },
    Case {
        name: "Dropout",
        build: || Box::new(Dropout::new(0.5)),
        input: Input::Float(&[4, 16]),
        check: Check::DeviceOnly,
    },
    Case {
        name: "Embedding",
        build: || Box::new(Embedding::new(20, 8)),
        input: Input::Indices,
        check: Check::Full,
    },
    Case {
        name: "LSTMCell (2 steps)",
        build: || Box::new(TwoSteps(LSTMCell::new(8, 6, true))),
        input: Input::Float(&[3, 8]),
        check: Check::Full,
    },
    Case {
        name: "MultiHeadAttention",
        build: || Box::new(MultiHeadAttention::new(16, 4, 0.0)),
        input: Input::Float(&[2, 5, 16]),
        check: Check::Full,
    },
    Case {
        name: "MultiHeadAttention (causal)",
        build: || Box::new(CausalMha(MultiHeadAttention::new(16, 4, 0.0))),
        input: Input::Float(&[2, 5, 16]),
        check: Check::Full,
    },
    Case {
        name: "FeedForward",
        build: || Box::new(FeedForward::new(16, 2, 0.0)),
        input: Input::Float(&[2, 5, 16]),
        check: Check::Full,
    },
    Case {
        name: "TransformerBlock",
        build: || Box::new(TransformerBlock::new(16, 4, 2, 0.0)),
        input: Input::Float(&[2, 5, 16]),
        check: Check::Full,
    },
    Case {
        name: "PositionalEncoding (learned)",
        build: || {
            Box::new(PositionalEncoding::new(
                16,
                8,
                PositionalEncodingType::Learned,
            ))
        },
        input: Input::Float(&[2, 5, 16]),
        check: Check::Full,
    },
];

/// The case's CPU input and its independent GPU copy.
fn inputs(input: Input) -> (Tensor, Tensor) {
    match input {
        Input::Float(shape) => {
            let x = RawTensor::randn(shape);
            x.set_requires_grad(true);
            let g = gpu_leaf(&x);
            (x, g)
        }
        Input::Indices => {
            let x = RawTensor::from_typed(vec![1i64, 3, 3, 19, 0, 7], &[6]);
            (x.clone(), x)
        }
    }
}

/// Compare `cpu` with `gpu` and push a message to `errs` for the first mismatch.
fn compare(errs: &mut Vec<String>, case: &str, what: &str, cpu: &[f32], gpu: &[f32]) {
    if cpu.len() != gpu.len() {
        errs.push(format!(
            "{case}: {what} length {} vs {}",
            cpu.len(),
            gpu.len()
        ));
        return;
    }
    let scale = cpu.iter().fold(1.0f32, |m, v| m.max(v.abs()));
    if let Some((i, (c, g))) = cpu
        .iter()
        .zip(gpu)
        .enumerate()
        .find(|(_, (c, g))| (*c - *g).abs() > TOL_MATMUL_BWD * scale || !g.is_finite())
    {
        errs.push(format!("{case}: {what}[{i}] cpu {c} gpu {g}"));
    }
}

/// Run `case` once on each device and return every problem found.
fn run(case: &Case) -> Vec<String> {
    let mut errs = Vec::new();
    manual_seed(0x464);
    let cpu = (case.build)();
    let mut gpu = (case.build)();
    gpu.load_state_dict(&cpu.state_dict())
        .expect("same architecture");
    gpu.to_device(gpu_device());
    let (x_cpu, x_gpu) = inputs(case.input);

    let out_cpu = cpu.forward(&x_cpu);
    let w_cpu = RawTensor::randn(out_cpu.shape().truncate_rank());
    let w_gpu = gpu_leaf(&w_cpu);
    out_cpu
        .elem_mul(&w_cpu)
        .sum()
        .into_loss()
        .unwrap()
        .backward();
    let mut opt_cpu = Adam::new(cpu.parameters(), 1e-2, (0.9, 0.999), 1e-8, 0.0);
    opt_cpu.step();

    let _strict = with_device_policy(DevicePolicy::Strict);
    let before = volta::gpu::gpu_counters();
    let out_gpu = gpu.forward(&x_gpu);
    out_gpu
        .elem_mul(&w_gpu)
        .sum()
        .into_loss()
        .unwrap()
        .backward();
    let mut opt_gpu = Adam::new(gpu.parameters(), 1e-2, (0.9, 0.999), 1e-8, 0.0);
    opt_gpu.step();
    let used = volta::gpu::gpu_counters().since(&before);
    // Adam leaves the gradients as they were; reading them is outside the window.
    let grads_gpu: Vec<Option<Vec<f32>>> = gpu.parameters().iter().map(Tensor::grad).collect();
    // The step stays on the device: no readback and no queue drain from the first forward
    // op to the end of the optimizer step. Uploads are allowed: the backward seed and the
    // few constants layers build with `constant_like` are small host-to-GPU copies, which
    // queue without waiting.
    if used.readbacks != 0 || used.syncs != 0 {
        errs.push(format!(
            "{}: the step waited on the host: {used:?}",
            case.name
        ));
    }

    if !out_gpu.device().is_gpu() {
        errs.push(format!("{}: output left the GPU", case.name));
    }
    for (i, p) in gpu.parameters().iter().enumerate() {
        if !p.device().is_gpu() {
            errs.push(format!("{}: parameter {i} left the GPU", case.name));
        }
    }
    if case.check == Check::DeviceOnly {
        return errs;
    }

    compare(
        &mut errs,
        case.name,
        "output",
        &out_cpu.data(),
        &out_gpu.data(),
    );
    if let Input::Float(_) = case.input {
        match (x_cpu.grad(), x_gpu.grad()) {
            (Some(c), Some(g)) => compare(&mut errs, case.name, "input grad", &c, &g),
            (c, g) => errs.push(format!(
                "{}: input grad present cpu {} gpu {}",
                case.name,
                c.is_some(),
                g.is_some()
            )),
        }
    }
    let params_cpu = cpu.parameters();
    for (i, (p, g)) in params_cpu.iter().zip(&grads_gpu).enumerate() {
        let what = format!("param {i} grad");
        match (p.grad(), g) {
            (Some(c), Some(g)) => {
                compare(&mut errs, case.name, &what, &c, g);
            }
            (c, g) => errs.push(format!(
                "{}: {what} present cpu {} gpu {}",
                case.name,
                c.is_some(),
                g.is_some()
            )),
        }
    }
    for (i, (c, g)) in params_cpu.iter().zip(&gpu.parameters()).enumerate() {
        compare(
            &mut errs,
            case.name,
            &format!("param {i} after step"),
            &c.data(),
            &g.data(),
        );
    }
    errs
}

/// Every built-in layer trains one step on the GPU under `Strict`, matching the CPU.
#[test]
fn every_layer_trains_one_step_on_gpu() {
    if skip_if_no_gpu() {
        return;
    }
    let errs: Vec<String> = CASES
        .iter()
        .flat_map(|case| {
            catch_unwind(AssertUnwindSafe(|| run(case))).unwrap_or_else(|panic| {
                let msg = panic
                    .downcast_ref::<String>()
                    .map(String::as_str)
                    .or_else(|| panic.downcast_ref::<&str>().copied())
                    .unwrap_or("non-string panic");
                vec![format!("{}: panicked: {msg}", case.name)]
            })
        })
        .collect();
    assert!(errs.is_empty(), "\n{}", errs.join("\n"));
}
