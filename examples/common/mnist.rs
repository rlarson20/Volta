//! Shared by the MNIST examples: what to print when `data/mnist` is missing, so the
//! instructions exist once and match the `just mnist-data` recipe (same mirror, all four
//! files, SHA-256 checked there), and a loader for one split (`train` or `t10k`).

use std::path::Path;
use volta::data::{load_mnist_images, load_mnist_labels, to_one_hot};

/// One MNIST split: pixels in `[0, 1]`, one-hot labels, and the sample count.
#[allow(
    dead_code,
    reason = "used by mnist_cnn and mnist_rnn; vae and dcgan include this module for the hint only"
)]
pub struct Split {
    /// `n * 28 * 28` pixels, scaled to `[0, 1]` by `load_mnist_images`.
    pub images: Vec<f32>,
    /// `n * 10` one-hot labels.
    pub labels: Vec<f32>,
    /// Number of samples.
    pub len: usize,
}

/// Load `<dir>/<prefix>-images-idx3-ubyte` and `<dir>/<prefix>-labels-idx1-ubyte`
/// (`prefix` is `train` or `t10k`). If either file is missing, print the download hint
/// and exit, rather than train or test on made-up data.
///
/// Asserts the pixels span exactly `[0, 1]`: a second rescale would stay inside the
/// range but collapse it.
#[allow(
    dead_code,
    reason = "used by mnist_cnn and mnist_rnn; vae and dcgan include this module for the hint only"
)]
pub fn load_split(dir: &str, prefix: &str) -> Split {
    let images_path = format!("{dir}/{prefix}-images-idx3-ubyte");
    let labels_path = format!("{dir}/{prefix}-labels-idx1-ubyte");
    if !Path::new(&images_path).exists() || !Path::new(&labels_path).exists() {
        eprintln!("MNIST {prefix} files not found in {dir}.");
        print_download_hint();
        std::process::exit(1);
    }
    let images = load_mnist_images(&images_path).expect("Failed to load MNIST images");
    let raw_labels = load_mnist_labels(&labels_path).expect("Failed to load MNIST labels");
    let len = raw_labels.len();
    assert_eq!(
        images.len(),
        len * 28 * 28,
        "{prefix}: image and label counts disagree"
    );
    let labels = to_one_hot(&raw_labels, 10).expect("MNIST labels are digits 0-9");

    let (lo, hi) = images
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &x| {
            (lo.min(x), hi.max(x))
        });
    assert!(
        lo == 0.0 && hi == 1.0,
        "{prefix} images must span [0, 1], found [{lo}, {hi}]"
    );
    Split {
        images,
        labels,
        len,
    }
}

/// Print how to fetch MNIST into `data/mnist`, with `just` and without it.
pub fn print_download_hint() {
    eprintln!("\nTo download MNIST data (from the repository root):");
    eprintln!("  just mnist-data");
    eprintln!("\nWithout just (the same mirror; `just mnist-data` also checks SHA-256 sums):");
    eprintln!("  mkdir -p data/mnist && cd data/mnist");
    eprintln!(
        "  for f in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do"
    );
    eprintln!(
        "    curl -sSfL -O \"https://ossci-datasets.s3.amazonaws.com/mnist/$f.gz\" && gunzip -f \"$f.gz\""
    );
    eprintln!("  done");
}
