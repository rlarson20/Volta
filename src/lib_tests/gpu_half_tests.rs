//! GPU f16 and bf16 against the CPU in the same dtype (ADDENDUM §11 P8), for native f16
//! (where the device has `SHADER_F16`), f16 forced through the packed emulation, and bf16
//! (always packed): unary and binary ops, matmul, reductions, movement, convolutions,
//! casts, comparisons and `where`, SGD/Adam/Muon steps, and an autocast MLP; plus Bool
//! round trips and the dtypes the GPU refuses. Both sides compute in f32 and round each
//! output, so values agree to the last bit or a rounding step; bounds are in epsilons of
//! the dtype relative to `max(|cpu|, rms(cpu))`. Without an adapter every test returns.
#![cfg(feature = "gpu")]

use crate::amp::autocast;
use crate::gpu::set_f16_emulation;
use crate::nn::{Layer, Module};
use crate::ops::{BinaryOp, UnaryOp};
use crate::{
    Adam, Axis, Conv2d, ConvAlgo, DType, Device, Linear, Muon, PadValue, RawTensor, ReLU, SGD,
    Sequential, Storage, Tensor, is_gpu_available,
};
use std::sync::{Mutex, MutexGuard};

/// A tensor op under test.
type Op = fn(&Tensor) -> Tensor;

/// Serializes these tests: the f16 emulation switch is process-wide, so one test's mode
/// must not leak into another's while it runs.
static MODE_LOCK: Mutex<()> = Mutex::new(());

/// A half mode under test.
#[derive(Clone, Copy, Debug)]
struct Mode {
    /// For messages.
    name: &'static str,
    /// The tensor dtype.
    dtype: DType,
    /// Force f16 through the packed shaders.
    emulate: bool,
}

/// Every mode this device can run, with the lock held and the switch reset on drop.
fn modes() -> Option<(MutexGuard<'static, ()>, Vec<Mode>)> {
    if !is_gpu_available() {
        return None;
    }
    let guard = MODE_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let ctx = crate::gpu::get_gpu_context().expect("an adapter");
    let mut modes = Vec::new();
    if ctx.elem_for(DType::F16) == crate::gpu::shader_template::Elem::F16 {
        modes.push(Mode {
            name: "f16 native",
            dtype: DType::F16,
            emulate: false,
        });
    }
    modes.push(Mode {
        name: "f16 emulated",
        dtype: DType::F16,
        emulate: true,
    });
    modes.push(Mode {
        name: "bf16",
        dtype: DType::BF16,
        emulate: false,
    });
    Some((guard, modes))
}

/// Run `f` for every mode, with the emulation switch set for it.
fn each_mode(f: impl Fn(Mode)) {
    let Some((_guard, modes)) = modes() else {
        return;
    };
    for m in modes {
        set_f16_emulation(m.emulate);
        f(m);
    }
    set_f16_emulation(false);
}

/// The dtype's machine epsilon.
fn eps(dtype: DType) -> f64 {
    match dtype {
        DType::F16 => 1.0 / 1024.0,
        DType::BF16 => 1.0 / 128.0,
        other @ (DType::F32 | DType::F64 | DType::I32 | DType::I64 | DType::U8 | DType::Bool) => {
            panic!("no half epsilon for {other}")
        }
    }
}

/// The GPU device.
fn gpu() -> Device {
    Device::gpu().expect("an adapter")
}

/// Seeded values in `[-2, 2)` for `n` elements.
fn values(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            #[expect(clippy::cast_precision_loss, reason = "test values")]
            let u = (s >> 40) as f32 / (1u64 << 24) as f32;
            4.0f32.mul_add(u, -2.0)
        })
        .collect()
}

/// A CPU leaf and a GPU leaf of `dtype` with the same (rounded) values, both requiring
/// grad.
fn pair(data: Vec<f32>, shape: &[usize], dtype: DType) -> (Tensor, Tensor) {
    let cpu = RawTensor::new(data, shape, false).to_dtype(dtype).detach();
    let g = cpu.detach().to_device(gpu());
    cpu.set_requires_grad(true);
    g.set_requires_grad(true);
    (cpu, g)
}

/// Every element of `s`, decoded to f64.
fn decode(s: &Storage) -> Vec<f64> {
    s.to_f32_vec().into_iter().map(f64::from).collect()
}

/// Asserts the GPU values `got` match the CPU values `want` within `k` epsilons.
fn assert_close(label: &str, mode: Mode, got: &[f64], want: &[f64], k: f64) {
    assert_eq!(got.len(), want.len(), "{}: {label}: length", mode.name);
    #[expect(clippy::cast_precision_loss, reason = "small test tensors")]
    let rms = (want.iter().map(|w| w * w).sum::<f64>() / want.len() as f64).sqrt();
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let err = (g - w).abs() / (eps(mode.dtype) * w.abs().max(rms).max(f64::MIN_POSITIVE));
        assert!(
            err <= k,
            "{}: {label}[{i}]: gpu {g:e}, cpu {w:e} ({err:.2} eps > {k})",
            mode.name
        );
    }
}

/// Compare outputs (dtype, device, values) and, after a weighted-sum backward on both
/// sides, the gradients of each `(cpu, gpu)` input pair.
fn check(
    label: &str,
    mode: Mode,
    out: (&Tensor, &Tensor),
    inputs: &[(&Tensor, &Tensor)],
    k: (f64, f64),
) {
    let (oc, og) = out;
    assert_eq!(og.dtype(), oc.dtype(), "{}: {label}: dtype", mode.name);
    assert!(
        og.device().is_gpu(),
        "{}: {label}: the output stays on the GPU",
        mode.name
    );
    assert_close(
        &format!("{label}.out"),
        mode,
        &decode(&og.borrow().data),
        &decode(&oc.borrow().data),
        k.0,
    );
    let n = oc.borrow().data.len();
    let w = values(n, 99);
    let shape = oc.shape();
    let wc = RawTensor::new(w, shape.truncate_rank(), false).to_dtype(oc.dtype());
    let wg = wc.to_device(gpu());
    oc.elem_mul(&wc)
        .sum()
        .into_loss()
        .expect("scalar")
        .backward();
    og.elem_mul(&wg)
        .sum()
        .into_loss()
        .expect("scalar")
        .backward();
    for (j, (ic, ig)) in inputs.iter().enumerate() {
        let gc = ic.grad_storage().expect("cpu grad");
        let gg = ig.grad_storage().expect("gpu grad");
        assert_eq!(
            gg.dtype(),
            gc.dtype(),
            "{}: {label}.grad{j}: dtype",
            mode.name
        );
        assert!(gg.is_gpu(), "{}: {label}.grad{j}: on the GPU", mode.name);
        assert_close(
            &format!("{label}.grad{j}"),
            mode,
            &decode(&gg),
            &decode(&gc),
            k.1,
        );
    }
}

/// Every unary op with a GPU kernel (positive inputs for the ones that need them).
#[test]
fn gpu_half_unary_ops_match_cpu() {
    each_mode(|m| {
        for &op in UnaryOp::ALL {
            if !op.has_gpu_kernel() {
                continue;
            }
            let mut v = values(37, 1);
            if matches!(
                op,
                UnaryOp::Log | UnaryOp::Log2 | UnaryOp::Sqrt | UnaryOp::Recip
            ) {
                v.iter_mut().for_each(|x| *x = x.abs() + 0.25);
            }
            let (c, g) = pair(v, &[37], m.dtype);
            let (oc, og) = (RawTensor::unary_op(&c, op), RawTensor::unary_op(&g, op));
            check(&format!("{op:?}"), m, (&oc, &og), &[(&c, &g)], (1.0, 2.0));
        }
    });
}

/// The differentiable binary ops on equal shapes (the GPU kernels' case): bit-exact except
/// division.
#[test]
fn gpu_half_binary_ops_match_cpu() {
    each_mode(|m| {
        for op in [
            BinaryOp::Add,
            BinaryOp::Sub,
            BinaryOp::Mul,
            BinaryOp::Div,
            BinaryOp::Max,
        ] {
            let mut bv = values(33, 3);
            if op == BinaryOp::Div {
                bv.iter_mut().for_each(|x| *x = x.abs() + 0.5);
            }
            let (ac, ag) = pair(values(33, 2), &[3, 11], m.dtype);
            let (bc, bg) = pair(bv, &[3, 11], m.dtype);
            let (oc, og) = (
                RawTensor::binary_op(&ac, &bc, op),
                RawTensor::binary_op(&ag, &bg, op),
            );
            // Sums, differences, products and maxima of half values are exact in f32, so
            // both sides round the same value: bit-exact. WGSL allows f32 division 2.5 ulp
            // of error, so a quotient can round a step either way.
            let k = if op == BinaryOp::Div { 1.0 } else { 0.0 };
            check(
                &format!("{op:?}"),
                m,
                (&oc, &og),
                &[(&ac, &ag), (&bc, &bg)],
                (k, k),
            );
        }
    });
}

/// 2-D matmul (summation order differs, so a rounding step either way).
#[test]
fn gpu_half_matmul_matches_cpu() {
    each_mode(|m| {
        let (ac, ag) = pair(values(5 * 7, 4), &[5, 7], m.dtype);
        let (bc, bg) = pair(values(7 * 3, 5), &[7, 3], m.dtype);
        check(
            "matmul",
            m,
            (&ac.matmul(&bc), &ag.matmul(&bg)),
            &[(&ac, &ag), (&bc, &bg)],
            (1.0, 1.0),
        );
    });
}

/// Full sum, mean and max, forward and backward.
#[test]
fn gpu_half_reductions_match_cpu() {
    each_mode(|m| {
        let ops: [(&str, Op); 3] = [
            ("sum", Tensor::sum),
            ("mean", Tensor::mean),
            ("max", Tensor::max_reduce),
        ];
        for (name, op) in ops {
            let (c, g) = pair(values(50, 6), &[50], m.dtype);
            check(name, m, (&op(&c), &op(&g)), &[(&c, &g)], (1.0, 0.0));
        }
    });
}

/// Movement ops copy values exactly, forward and backward.
#[test]
fn gpu_half_movement_matches_cpu() {
    each_mode(|m| {
        let ops: [(&str, Op); 4] = [
            ("permute", |t| t.permute(&[Axis(2), Axis(0), Axis(1)])),
            ("expand", |t| {
                t.shrink(&[(0, 2), (0, 1), (0, 4)]).expand(&[2, 5, 4])
            }),
            ("shrink", |t| t.shrink(&[(0, 2), (1, 3), (1, 3)])),
            ("pad", |t| {
                t.pad_with(&[(0, 0), (1, 0), (0, 2)], PadValue::ZERO)
            }),
        ];
        for (name, op) in ops {
            let (c, g) = pair(values(24, 7), &[2, 3, 4], m.dtype);
            check(name, m, (&op(&c), &op(&g)), &[(&c, &g)], (0.0, 1.0));
        }
    });
}

/// `Conv2d` with each algorithm, input and parameters in the half dtype on both sides.
#[test]
fn gpu_half_conv2d_matches_cpu() {
    each_mode(|m| {
        for algo in [ConvAlgo::Direct, ConvAlgo::Im2col, ConvAlgo::IGEMM] {
            // Same seed, same weights: half models can't round-trip a state dict yet.
            let build = || {
                crate::manual_seed(11);
                let mut c = Conv2d::new(2, 3, 3, 1, 1, true);
                c.set_algo(algo);
                c.to_dtype(m.dtype);
                c
            };
            let cpu = build();
            let mut gpu_conv = build();
            gpu_conv.to_device(gpu());
            let (xc, xg) = pair(values(2 * 2 * 5 * 5, 8), &[2, 2, 5, 5], m.dtype);
            let (oc, og) = (cpu.forward(&xc), gpu_conv.forward(&xg));
            check(
                &format!("conv {algo:?}"),
                m,
                (&oc, &og),
                &[(&xc, &xg)],
                (2.0, 2.0),
            );
            for (pc, pg) in cpu.parameters().iter().zip(gpu_conv.parameters()) {
                let (gc, gg) = (
                    pc.grad_storage().expect("grad"),
                    pg.grad_storage().expect("grad"),
                );
                assert_eq!(gg.dtype(), m.dtype);
                assert_close(
                    &format!("conv {algo:?} param grad"),
                    m,
                    &decode(&gg),
                    &decode(&gc),
                    2.0,
                );
            }
        }
    });
}

/// Casts between the GPU floats and Bool, comparisons and `where`, exactly.
#[test]
fn gpu_half_cast_compare_where_match_cpu_exactly() {
    each_mode(|m| {
        let mut v = values(20, 9);
        *v.get_mut(3).expect("20 values") = 0.0;
        *v.get_mut(4).expect("20 values") = 1e-6; // below f16's normal range: a subnormal or 0
        let c = RawTensor::new(v.clone(), &[20], false);
        let g = c.to_device(gpu());
        for target in [m.dtype, DType::F32, DType::Bool] {
            let (hc, hg) = (
                c.to_dtype(m.dtype).to_dtype(target),
                g.to_dtype(m.dtype).to_dtype(target),
            );
            assert!(hg.device().is_gpu());
            assert_eq!(
                hg.to_device(Device::CPU).data(),
                hc.data(),
                "{}: cast to {target}",
                m.name
            );
        }
        let (a, b) = (
            c.to_dtype(m.dtype),
            RawTensor::new(values(20, 10), &[20], false).to_dtype(m.dtype),
        );
        let (ag, bg) = (a.to_device(gpu()), b.to_device(gpu()));
        for (name, cpu, gpu_t) in [
            ("lt", a.lt(&b), ag.lt(&bg)),
            ("le", a.le(&a), ag.le(&ag)),
            ("gt", a.gt(&b), ag.gt(&bg)),
            ("ge", a.ge(&b), ag.ge(&bg)),
            ("eq", a.elem_eq(&b), ag.elem_eq(&bg)),
            ("ne", a.elem_ne(&a), ag.elem_ne(&ag)),
        ] {
            assert_eq!(gpu_t.dtype(), DType::Bool);
            assert!(
                gpu_t.device().is_gpu(),
                "{}: {name} stays on the GPU",
                m.name
            );
            assert_eq!(
                gpu_t
                    .to_device(Device::CPU)
                    .to_vec_as::<bool>()
                    .expect("bool"),
                cpu.to_vec_as::<bool>().expect("bool"),
                "{}: {name}",
                m.name
            );
        }
        let cond = a.lt(&b);
        let (wc, wg) = (
            cond.where_op(&a, &b),
            cond.to_device(gpu()).where_op(&ag, &bg),
        );
        assert!(wg.device().is_gpu());
        assert_eq!(
            wg.to_device(Device::CPU).data(),
            wc.data(),
            "{}: where",
            m.name
        );
    });
}

/// SGD with momentum, Adam and Muon step a half parameter on the GPU as on the CPU.
#[test]
fn gpu_half_optimizers_match_cpu() {
    each_mode(|m| {
        for name in ["sgd", "adam", "muon"] {
            let (pc, pg) = pair(values(12, 12), &[3, 4], m.dtype);
            let mut step: Vec<Box<dyn FnMut()>> = Vec::new();
            for p in [pc.clone(), pg.clone()] {
                step.push(match name {
                    "sgd" => {
                        let mut o = SGD::new(vec![p], 0.0625, 0.875, 0.0);
                        Box::new(move || o.step())
                    }
                    "adam" => {
                        let mut o = Adam::new(vec![p], 0.007_812_5, (0.875, 0.999), 1e-4, 0.0);
                        Box::new(move || o.step())
                    }
                    _ => {
                        let mut o = Muon::new(vec![p], 0.02, 0.95, true, 5);
                        Box::new(move || o.step())
                    }
                });
            }
            for k in 0..3 {
                let g = RawTensor::new(values(12, 20 + k), &[3, 4], false)
                    .to_dtype(m.dtype)
                    .borrow()
                    .data
                    .clone();
                pc.set_grad(g.clone());
                pg.set_grad(g);
                for s in &mut step {
                    s();
                }
                assert_eq!(pg.dtype(), m.dtype);
                assert!(pg.device().is_gpu());
                assert_close(
                    &format!("{name} step {k}"),
                    m,
                    &decode(&pg.borrow().data),
                    &decode(&pc.borrow().data),
                    2.0,
                );
            }
        }
    });
}

/// An f32 MLP under autocast on the GPU matches the same model under autocast on the CPU:
/// the f32 loss and every f32 parameter gradient.
#[test]
fn gpu_autocast_mlp_matches_cpu() {
    each_mode(|m| {
        crate::manual_seed(21);
        let cpu = Sequential::new(vec![
            Box::new(Linear::new(6, 8, true)),
            Box::new(ReLU),
            Box::new(Linear::new(8, 3, true)),
        ]);
        let mut gpu_model = Sequential::new(vec![
            Box::new(Linear::new(6, 8, true)),
            Box::new(ReLU),
            Box::new(Linear::new(8, 3, true)),
        ]);
        gpu_model
            .load_state_dict(&cpu.state_dict())
            .expect("same shapes");
        gpu_model.to_device(gpu());
        let x = RawTensor::new(values(4 * 6, 22), &[4, 6], false);
        let xg = x.to_device(gpu());
        let y = RawTensor::new(values(4 * 3, 23), &[4, 3], false);
        let yg = y.to_device(gpu());
        // `Linear`'s bias add broadcasts, which has no GPU kernel in any dtype (an existing
        // f32 gap): run it under `PreferGpu`, which computes that add on the CPU and moves
        // it back.
        let _prefer = crate::with_device_policy(crate::DevicePolicy::PreferGpu);
        let (lc, lg) = {
            let _ac = autocast(m.dtype);
            (
                RawTensor::mse_loss(&cpu.forward(&x), &y),
                RawTensor::mse_loss(&gpu_model.forward(&xg), &yg),
            )
        };
        assert_eq!((lc.dtype(), lg.dtype()), (DType::F32, DType::F32));
        assert_close(
            "loss",
            m,
            &[f64::from(lg.item())],
            &[f64::from(lc.item())],
            2.0,
        );
        lc.into_loss().expect("scalar").backward();
        lg.into_loss().expect("scalar").backward();
        for (pc, pg) in cpu.parameters().iter().zip(gpu_model.parameters()) {
            let (gc, gg) = (
                pc.grad_storage().expect("grad"),
                pg.grad_storage().expect("grad"),
            );
            assert_eq!(gg.dtype(), DType::F32, "f32 parameters get f32 gradients");
            assert!(gg.is_gpu());
            assert_close("param grad", m, &decode(&gg), &decode(&gc), 4.0);
        }
    });
}

/// Bool tensors round-trip through the GPU exactly.
#[test]
fn gpu_bool_round_trips() {
    if !is_gpu_available() {
        return;
    }
    let pattern = vec![true, false, false, true, true];
    let t = RawTensor::from_typed(pattern.clone(), &[5]);
    let g = t.to_device(gpu());
    assert_eq!(g.dtype(), DType::Bool);
    assert_eq!(
        g.to_device(Device::CPU).to_vec_as::<bool>().expect("bool"),
        pattern
    );
}

/// The GPU refuses f64 (as PyTorch's MPS does) and integer tensors, naming the cast.
#[test]
fn gpu_refuses_f64_and_integer_tensors() {
    if !is_gpu_available() {
        return;
    }
    for (t, name) in [
        (RawTensor::from_typed(vec![1.0f64], &[1]), "f64"),
        (RawTensor::from_typed(vec![1i64], &[1]), "i64"),
    ] {
        let err = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| t.to_device(gpu())))
            .expect_err("refused");
        let msg = err.downcast_ref::<String>().map_or("", String::as_str);
        assert!(
            msg.contains(&format!(
                "GPU tensors hold f32, f16, bf16 or bool, not {name}"
            )),
            "{msg}"
        );
    }
}

/// Prints which half modes this device runs (`--nocapture`), so a log shows whether the
/// native f16 path was exercised or only the emulated one.
#[test]
fn gpu_half_modes_report() {
    if let Some((_guard, modes)) = modes() {
        let names: Vec<&str> = modes.iter().map(|m| m.name).collect();
        println!("GPU half modes: {names:?}");
    }
}

/// `GradScaler` on the GPU under f16 autocast: a loss scaled by 2e-8 keeps its gradients
/// (unscaled on the device by the fused kernel) within f16 precision of the CPU f32 run,
/// and an inf gradient on a GPU parameter skips the step and halves the scale.
#[test]
fn gpu_grad_scaler_unscales_on_the_device() {
    let Some((_guard, _modes)) = modes() else {
        return;
    };
    let _prefer = crate::with_device_policy(crate::DevicePolicy::PreferGpu);
    let tiny = 2e-8f32;
    let build = || {
        crate::manual_seed(31);
        Linear::new(4, 2, true)
    };
    let x = RawTensor::new(values(8 * 4, 32), &[8, 4], false);
    let y = RawTensor::new(values(8 * 2, 33), &[8, 2], false);

    let reference = build();
    RawTensor::mse_loss(&reference.forward(&x), &y)
        .elem_mul(&RawTensor::new(vec![tiny], &[1], false))
        .into_loss()
        .expect("scalar")
        .backward();

    let mut model = build();
    model.to_device(gpu());
    let (xg, yg) = (x.to_device(gpu()), y.to_device(gpu()));
    let mut opt = SGD::new(model.parameters(), 0.0, 0.0, 0.0);
    let mut scaler = crate::GradScaler::new();
    let loss = {
        let _ac = autocast(DType::F16);
        RawTensor::mse_loss(&model.forward(&xg), &yg)
    };
    let loss = loss.elem_mul(&RawTensor::new(vec![tiny], &[1], false).to_device(gpu()));
    scaler.scale(&loss).into_loss().expect("scalar").backward();
    scaler.unscale(&opt);
    for (p, r) in model.parameters().iter().zip(reference.parameters()) {
        let g = p.grad_storage().expect("grad");
        assert!(g.is_gpu(), "the unscaled gradient stays on the GPU");
        let (got, want) = (g.to_f32_vec(), r.grad().expect("grad"));
        assert_eq!(got.len(), want.len());
        for (&gv, &wv) in got.iter().zip(&want) {
            assert!(gv != 0.0, "the scaled gradient survived f16");
            assert!(
                (gv - wv).abs() <= 4e-3 * wv.abs(),
                "gpu {gv:e} vs f32 {wv:e}"
            );
        }
    }
    assert!(scaler.step(&mut opt), "finite gradients step");
    scaler.update();

    let p = model.parameters().into_iter().next().expect("a weight");
    let before = p.data();
    let mut bad = vec![0.0f32; before.len()];
    *bad.first_mut().expect("elements") = f32::INFINITY;
    p.set_grad(Storage::cpu(bad));
    assert!(
        !scaler.step(&mut opt),
        "an inf gradient on the GPU skips the step"
    );
    scaler.update();
    assert_eq!(p.data(), before, "parameters unchanged");
    assert_eq!(scaler.get_scale(), 32768.0);
}
