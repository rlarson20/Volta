//! Shared by the MNIST examples: what to print when `data/mnist` is missing, so the
//! instructions exist once and match the `just mnist-data` recipe (same mirror, all four
//! files, SHA-256 checked there).

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
