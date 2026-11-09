//! MNIST under bf16 autocast against f32 (ADDENDUM §11 P7's done-criterion).
//!
//! Trains the same MLP (784 → 256 → `ReLU` → 10, Adam lr 1e-3, batch 64, 3 epochs) twice per
//! seed, once in f32 and once with its forward under `autocast(DType::BF16)`, from the same
//! initialization and data order, for 5 seeds. It prints each run's test accuracy and
//! exits 1 unless the bf16 mean is within 0.2 percentage points of the f32 mean.
//!
//! The margin was fixed before autocast existed, from measurements (DECISIONS.md, P7):
//! seed-to-seed spread is about 0.25 points, so the mean of 5 has a standard error near
//! 0.05; PyTorch's own f32-to-autocast gap on this MLP is 0.01 points; and training a model
//! stored in bf16 (no f32 parameters) loses 0.39 points, which this check rejects.
//!
//! Needs the MNIST files under `data/mnist`: `just mnist-data` downloads and checks them,
//! and `just mnist-autocast` runs this in release (about 3 minutes on one core).

use std::path::Path;
use std::time::Instant;
use volta::amp::autocast;
use volta::{
    Adam, DType, Layer, Linear, Module, RawTensor, ReLU, Sequential, Tensor, load_mnist_images,
    load_mnist_labels,
};

/// Seeds per precision.
const SEEDS: [u64; 5] = [1, 2, 3, 4, 5];
/// Training epochs per run.
const EPOCHS: usize = 3;
/// Minibatch size.
const BATCH: usize = 64;
/// The largest accepted gap between the f32 and bf16 mean test accuracies, in points.
const MARGIN: f64 = 0.2;
/// Pixels per image.
const PIXELS: usize = 784;

/// Images (flattened, in [0, 1]) and labels.
struct Split {
    /// `n * 784` pixels.
    images: Vec<f32>,
    /// `n` labels.
    labels: Vec<u8>,
}

impl Split {
    /// Load `<dir>/<prefix>-images-idx3-ubyte` and the matching labels.
    fn load(dir: &str, prefix: &str) -> Self {
        let images = load_mnist_images(format!("{dir}/{prefix}-images-idx3-ubyte"))
            .expect("readable MNIST images");
        let labels = load_mnist_labels(format!("{dir}/{prefix}-labels-idx1-ubyte"))
            .expect("readable MNIST labels");
        assert_eq!(images.len(), labels.len() * PIXELS, "{prefix}: image count");
        Self { images, labels }
    }

    /// The batch of samples `idx`: images `[b, 784]` (f32) and labels `[b]` (I64).
    fn batch(&self, idx: &[usize]) -> (Tensor, Tensor) {
        let mut x = Vec::with_capacity(idx.len() * PIXELS);
        let mut y = Vec::with_capacity(idx.len());
        for &i in idx {
            x.extend_from_slice(
                self.images
                    .get(i * PIXELS..(i + 1) * PIXELS)
                    .expect("index within the split"),
            );
            y.push(i64::from(
                *self.labels.get(i).expect("index within the split"),
            ));
        }
        (
            RawTensor::new(x, &[idx.len(), PIXELS], false),
            RawTensor::from_typed(y, &[idx.len()]),
        )
    }
}

/// `0..n` shuffled by a seeded LCG (the same order for both precisions).
fn shuffled(n: usize, seed: u64) -> Vec<usize> {
    let mut order: Vec<usize> = (0..n).collect();
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    for i in (1..n).rev() {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let j = usize::try_from(s >> 33).expect("31 bits fit usize") % (i + 1);
        order.swap(i, j);
    }
    order
}

/// Run `f` under `autocast(low)` if `low` is set, else plainly.
fn maybe_autocast<R>(low: Option<DType>, f: impl FnOnce() -> R) -> R {
    let _ac = low.map(autocast);
    f()
}

/// Train one MLP and return its test accuracy in percent.
fn run(train: &Split, test: &Split, seed: u64, low: Option<DType>) -> f64 {
    volta::manual_seed(seed);
    let model = Sequential::new(vec![
        Box::new(Linear::new(PIXELS, 256, true)),
        Box::new(ReLU),
        Box::new(Linear::new(256, 10, true)),
    ]);
    let mut opt = Adam::new(model.parameters(), 1e-3, (0.9, 0.999), 1e-8, 0.0);
    for epoch in 0..EPOCHS {
        let order = shuffled(train.labels.len(), seed * 1000 + epoch as u64);
        for chunk in order.chunks(BATCH) {
            let (x, y) = train.batch(chunk);
            opt.zero_grad();
            let loss = maybe_autocast(low, || {
                RawTensor::cross_entropy_loss(&model.forward(&x), &y)
            });
            loss.into_loss().expect("a scalar loss").backward();
            opt.step();
        }
    }

    let _no_grad = volta::autograd::no_grad();
    let mut correct = 0usize;
    let all: Vec<usize> = (0..test.labels.len()).collect();
    for chunk in all.chunks(1000) {
        let (x, y) = test.batch(chunk);
        let logits = maybe_autocast(low, || model.forward(&x)).to_dtype(DType::F32);
        let logits = logits.data();
        let labels = y.to_vec_as::<i64>().expect("I64 labels");
        for (r, &label) in labels.iter().enumerate() {
            let row = logits
                .get(r * 10..(r + 1) * 10)
                .expect("ten logits per row");
            let best = (0..10)
                .max_by(|&a, &b| {
                    row.get(a)
                        .expect("class")
                        .total_cmp(row.get(b).expect("class"))
                })
                .expect("ten classes");
            if i64::try_from(best).expect("class fits i64") == label {
                correct += 1;
            }
        }
    }
    #[expect(clippy::cast_precision_loss, reason = "counts below 2^24")]
    let acc = 100.0 * correct as f64 / test.labels.len() as f64;
    acc
}

/// The mean of `v`.
fn mean(v: &[f64]) -> f64 {
    #[expect(clippy::cast_precision_loss, reason = "five values")]
    let n = v.len() as f64;
    v.iter().sum::<f64>() / n
}

fn main() {
    let dir = "data/mnist";
    if !Path::new(&format!("{dir}/train-images-idx3-ubyte")).exists() {
        eprintln!("MNIST data not found in {dir}; run `just mnist-data` first.");
        std::process::exit(1);
    }
    let train = Split::load(dir, "train");
    let test = Split::load(dir, "t10k");
    let (lo, hi) = train
        .images
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &x| {
            (lo.min(x), hi.max(x))
        });
    assert!(
        (lo, hi) == (0.0, 1.0),
        "pixels must span [0, 1], got [{lo}, {hi}]"
    );

    println!("MNIST MLP, {EPOCHS} epochs, batch {BATCH}: test accuracy (%)");
    println!("{:>6} {:>8} {:>10} {:>8}", "seed", "f32", "bf16 amp", "gap");
    let (mut f32s, mut bf16s) = (Vec::new(), Vec::new());
    for seed in SEEDS {
        let t0 = Instant::now();
        let a = run(&train, &test, seed, None);
        let b = run(&train, &test, seed, Some(DType::BF16));
        println!(
            "{seed:>6} {a:>8.2} {b:>10.2} {:>8.2}   ({:.0} s)",
            a - b,
            t0.elapsed().as_secs_f64()
        );
        f32s.push(a);
        bf16s.push(b);
    }
    let gap = mean(&f32s) - mean(&bf16s);
    println!(
        "{:>6} {:>8.2} {:>10.2} {gap:>8.2}",
        "mean",
        mean(&f32s),
        mean(&bf16s)
    );
    if gap.abs() > MARGIN {
        eprintln!("FAIL: |mean gap| {:.2} > {MARGIN} points", gap.abs());
        std::process::exit(1);
    }
    println!("OK: |mean gap| {:.2} <= {MARGIN} points", gap.abs());
}
