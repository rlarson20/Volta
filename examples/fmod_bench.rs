//! Throwaway benchmark for #355 (not for merge): times GPU `modulo` with whichever
//! `floored_mod` `elementwise.wgsl` holds, against the CPU fallback a device policy would take
//! (download both operands, CPU `modulo`, upload the result) and against the check that would
//! decide whether to fall back (`max((a/b)^2)` reduced to a scalar and read back).
//! Also counts bitwise mismatches against the CPU result.
#![allow(
    clippy::unreadable_literal,
    clippy::print_stdout,
    clippy::cast_precision_loss,
    clippy::unwrap_used
)]
use std::time::Instant;
use volta::{Device, RawTensor, Tensor};

/// A seeded LCG, so both kernels see the same inputs.
struct Lcg(u64);
impl Lcg {
    /// Uniform in [0, 1).
    fn unit(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 40) as f32) / ((1u64 << 24) as f32)
    }
    /// Sign-random value with magnitude log-uniform in [10^lo, 10^hi].
    fn logu(&mut self, lo: f32, hi: f32, signed: bool) -> f32 {
        let m = 10f32.powf(lo + (hi - lo) * self.unit());
        if signed && self.unit() < 0.5 { -m } else { m }
    }
}

/// The median of `v`.
fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v.get(v.len() / 2).copied().expect("non-empty")
}

/// Median wall time of `f` in ms over `reps` runs, after three warm-up runs.
fn time<F: FnMut()>(reps: usize, mut f: F) -> f64 {
    for _ in 0..3 {
        f();
    }
    let mut ts = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t0 = Instant::now();
        f();
        ts.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    median(ts)
}

fn main() {
    let Some(gpu) = Device::gpu() else {
        println!("No GPU");
        return;
    };
    let n: usize = std::env::var("N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1 << 22);
    let reps: usize = std::env::var("REPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(15);
    let label = std::env::var("LABEL").unwrap_or_default();
    println!("GPU {gpu}, n = {n}, reps = {reps}, kernel = {label}");
    // (name, a magnitudes, b magnitudes): quotient exponent gap d ~ 3.3 * (a_hi - b_lo).
    let cases = [
        ("small |a/b| < 1e3", (-2.0, 2.0), (-1.0, 1.0)),
        ("|a/b| ~ 1e10 (past 2^23)", (9.0, 11.0), (0.0, 1.0)),
        (
            "worst: |a| ~ 1e38, |b| ~ 1e-37",
            (37.0, 38.0),
            (-37.0, -36.0),
        ),
    ];
    {
        let sp = [
            0.0f32,
            -0.0,
            1e-45,
            -1e-45,
            1.1754942e-38,
            1.1754944e-38,
            3.0e-39,
            1.0,
            -1.0,
            7.0,
            -7.0,
            3.4028235e38,
            -3.4028235e38,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            1e10,
            67_108_864.0,
            0.1,
            2.5,
        ];
        let (mut av, mut bv) = (Vec::new(), Vec::new());
        for &x in &sp {
            for &y in &sp {
                av.push(x);
                bv.push(y);
            }
        }
        let m = av.len();
        let (ac, bc) = (
            RawTensor::new(av.clone(), &[m], false),
            RawTensor::new(bv.clone(), &[m], false),
        );
        let want = ac.modulo(&bc).data();
        let got = ac
            .to_device(gpu.clone())
            .modulo(&bc.to_device(gpu.clone()))
            .data();
        let mut bad = 0;
        for i in 0..m {
            let (w, g) = (
                want.get(i).copied().expect("len"),
                got.get(i).copied().expect("len"),
            );
            if w.to_bits() != g.to_bits() && !(w.is_nan() && g.is_nan()) {
                bad += 1;
                if bad <= 12 {
                    println!(
                        "  special mismatch: {:e} mod {:e}: cpu {:e} gpu {:e}",
                        av.get(i).copied().expect("len"),
                        bv.get(i).copied().expect("len"),
                        w,
                        g
                    );
                }
            }
        }
        println!(
            "[{label}] specials (zeros, subnormals, max, inf, NaN; all pairs): mismatches {bad}/{m}"
        );
    }
    for (name, (alo, ahi), (blo, bhi)) in cases {
        let mut rng = Lcg(0x5eed);
        let a: Vec<f32> = (0..n).map(|_| rng.logu(alo, ahi, true)).collect();
        let b: Vec<f32> = (0..n).map(|_| rng.logu(blo, bhi, true)).collect();
        let a_cpu = RawTensor::new(a, &[n], false);
        let b_cpu = RawTensor::new(b, &[n], false);
        let want = a_cpu.modulo(&b_cpu).data();
        let a_g: Tensor = a_cpu.to_device(gpu.clone());
        let b_g: Tensor = b_cpu.to_device(gpu.clone());

        let got = a_g.modulo(&b_g).data();
        let mism = want
            .iter()
            .zip(&got)
            .filter(|(w, g)| w.to_bits() != g.to_bits() && !(w.is_nan() && g.is_nan()))
            .count();

        let t_gpu = time(reps, || {
            let _c = a_g.modulo(&b_g);
            assert!(volta::gpu::gpu_sync());
        });
        let t_check = time(reps, || {
            let q = a_g.div(&b_g);
            let _m = q.elem_mul(&q).max_reduce().item();
        });
        let t_fallback = time(reps, || {
            let a_h = a_g.to_device(Device::CPU);
            let b_h = b_g.to_device(Device::CPU);
            let _c = a_h.modulo(&b_h).to_device(gpu.clone());
            assert!(volta::gpu::gpu_sync());
        });
        let t_cpu_only = time(reps, || {
            let _c = a_cpu.modulo(&b_cpu);
        });
        println!(
            "[{label}] {name}: gpu mod {t_gpu:.3} ms | fallback check {t_check:.3} ms | \
             host round trip {t_fallback:.3} ms (cpu mod alone {t_cpu_only:.3} ms) | \
             mismatches vs CPU {mism}/{n}"
        );
    }
}
