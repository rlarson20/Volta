//! DCGAN on MNIST: a transposed-convolution `Generator` and a convolutional `Discriminator`, trained adversarially with `Adam` and `bce_with_logits_loss`.
//!
//! Needs the MNIST training images in `data/mnist/`; it exits with download instructions if they are missing, rather than train on made-up data.
//!
//! Images are normalized to `[-1, 1]` to match the generator's `Tanh` output. The example asserts that range before it trains, and exits if the losses go non-finite or the generator's output collapses to a constant.
use volta::{
    Adam, Conv2d, ConvTranspose2d, Flatten, Layer, Linear, Module, ProgressBar, RawTensor, Tanh,
    Tensor, bce_with_logits_loss, load_mnist_images, normalize, randn,
};

// Generator: noise -> fake images
/// Generator: latent noise to a fake 28x28 image.
struct Generator {
    /// Projects the latent vector to `256 * 7 * 7` features.
    fc: Linear,
    /// Upsamples `[256, 7, 7]` to `[128, 14, 14]`.
    deconv1: ConvTranspose2d,
    /// Upsamples `[128, 14, 14]` to `[1, 28, 28]`.
    deconv2: ConvTranspose2d,
    /// Squashes the output into `[-1, 1]`.
    tanh: Tanh,
}

impl Generator {
    /// Builds the generator for `latent_dim`-wide noise.
    fn new(latent_dim: usize) -> Self {
        Self {
            fc: Linear::new(latent_dim, 256 * 7 * 7, true),
            deconv1: ConvTranspose2d::new(256, 128, 4, 2, 1, true),
            deconv2: ConvTranspose2d::new(128, 1, 4, 2, 1, true),
            tanh: Tanh,
        }
    }

    /// Maps noise `[batch, latent_dim]` to images `[batch, 1, 28, 28]` in `[-1, 1]`.
    fn forward(&self, z: &volta::Tensor) -> volta::Tensor {
        let batch = z
            .shape()
            .dim(volta::Axis(0))
            .expect("noise has a batch dim");

        // z: [batch, latent_dim] -> [batch, 256*7*7]
        let x = self.fc.forward(z).relu();

        // Reshape: [batch, 256*7*7] -> [batch, 256, 7, 7]
        let x = x.reshape(&[batch, 256, 7, 7]);

        // [batch, 256, 7, 7] -> [batch, 128, 14, 14]
        let x = self.deconv1.forward(&x).relu();

        // [batch, 128, 14, 14] -> [batch, 1, 28, 28]
        let x = self.deconv2.forward(&x);

        // Tanh to get values in [-1, 1]
        self.tanh.forward(&x)
    }

    /// Trainable parameters of `fc`, `deconv1`, and `deconv2`.
    fn parameters(&self) -> Vec<volta::Tensor> {
        let mut params = Vec::new();
        params.extend(self.fc.parameters());
        params.extend(self.deconv1.parameters());
        params.extend(self.deconv2.parameters());
        params
    }
}

// Discriminator: images -> real/fake classification
/// Discriminator: a 28x28 image to a single real-vs-fake logit.
struct Discriminator {
    /// Downsamples `[1, 28, 28]` to `[64, 14, 14]`.
    conv1: Conv2d,
    /// Downsamples `[64, 14, 14]` to `[128, 7, 7]`.
    conv2: Conv2d,
    /// Flattens to `128 * 7 * 7` features.
    flatten: Flatten,
    /// Maps the features to one logit. There is no sigmoid; the loss is `bce_with_logits_loss`.
    fc: Linear,
}

impl Discriminator {
    /// Builds the discriminator.
    fn new() -> Self {
        Self {
            conv1: Conv2d::new(1, 64, 4, 2, 1, true),
            conv2: Conv2d::new(64, 128, 4, 2, 1, true),
            flatten: Flatten,
            fc: Linear::new(128 * 7 * 7, 1, true),
        }
    }

    /// Maps images `[batch, 1, 28, 28]` to logits `[batch, 1]`.
    fn forward(&self, x: &volta::Tensor) -> volta::Tensor {
        // [batch, 1, 28, 28] -> [batch, 64, 14, 14]
        let x = self.conv1.forward(x).relu();

        // [batch, 64, 14, 14] -> [batch, 128, 7, 7]
        let x = self.conv2.forward(&x).relu();

        // Flatten to [batch, 128*7*7]
        let x = self.flatten.forward(&x);

        // [batch, 128*7*7] -> [batch, 1]
        self.fc.forward(&x)
    }

    /// Trainable parameters of `conv1`, `conv2`, and `fc`.
    fn parameters(&self) -> Vec<volta::Tensor> {
        let mut params = Vec::new();
        params.extend(self.conv1.parameters());
        params.extend(self.conv2.parameters());
        params.extend(self.fc.parameters());
        params
    }
}

/// Mean and standard deviation over every element of `t`.
fn mean_std(t: &Tensor) -> (f32, f32) {
    let data = t.data();
    let n = data.len() as f32;
    let mean = data.iter().sum::<f32>() / n;
    let var = data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / n;
    (mean, var.sqrt())
}

fn main() {
    println!("=== DCGAN on MNIST ===\n");

    let latent_dim = 100;
    let batch_size = 64;
    let num_epochs = 5;
    let learning_rate = 0.0002;

    // Load MNIST data
    let train_images_path = "data/mnist/train-images-idx3-ubyte";
    let train_images = match load_mnist_images(train_images_path) {
        Ok(images) => images,
        Err(err) => {
            eprintln!("Could not load MNIST images from {train_images_path}: {err}");
            eprintln!("\nTo download MNIST data:");
            eprintln!("  mkdir -p data/mnist && cd data/mnist");
            eprintln!("  curl -O http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz");
            eprintln!("  gunzip *.gz");
            std::process::exit(1);
        }
    };
    let num_train = train_images.len() / 784;
    println!("Loaded {num_train} MNIST training images");

    // `load_mnist_images` already scales pixels to [0, 1]; map them to [-1, 1] to match tanh output.
    let mut train_data = train_images;
    normalize(&mut train_data, 0.5, 0.5);
    // MNIST contains both black (0) and white (255) pixels, so both ends of the range are hit.
    // Checking both ends catches a double normalization, which stays inside [-1, 1] but collapses.
    let (lo, hi) = train_data
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &x| {
            (lo.min(x), hi.max(x))
        });
    assert!(
        (lo + 1.0).abs() < 1e-6 && (hi - 1.0).abs() < 1e-6,
        "training images must span [-1, 1], found [{lo}, {hi}]"
    );

    println!("Architecture:");
    println!("  Generator:");
    println!("    Linear({} -> {}) -> ReLU", latent_dim, 256 * 7 * 7);
    println!("    Reshape -> [batch, 256, 7, 7]");
    println!("    ConvTranspose2d(256 -> 128) -> ReLU  [7x7 -> 14x14]");
    println!("    ConvTranspose2d(128 -> 1) -> Tanh    [14x14 -> 28x28]");
    println!("\n  Discriminator:");
    println!("    Conv2d(1 -> 64) -> ReLU   [28x28 -> 14x14]");
    println!("    Conv2d(64 -> 128) -> ReLU [14x14 -> 7x7]");
    println!("    Flatten -> Linear({} -> 1)", 128 * 7 * 7);
    println!("\nStarting training...\n");

    // Build models
    let generator = Generator::new(latent_dim);
    let discriminator = Discriminator::new();

    // The DCGAN paper's settings: lr 2e-4 and beta1 0.5 are for Adam.
    let betas = (0.5, 0.999);
    let mut g_optimizer = Adam::new(generator.parameters(), learning_rate, betas, 1e-8, 0.0);
    let mut d_optimizer = Adam::new(discriminator.parameters(), learning_rate, betas, 1e-8, 0.0);

    // A fixed noise batch, so the generator's output can be compared before and after training.
    let probe_noise = randn(&[16, latent_dim]);
    let (mean_before, std_before) = mean_std(&generator.forward(&probe_noise));
    println!("Generator output before training: mean={mean_before:.4}, std={std_before:.4}");

    let num_batches = num_train / batch_size;
    let mut progress = ProgressBar::new(num_epochs, "Training");

    for epoch in 0..num_epochs {
        let mut total_d_loss = 0.0;
        let mut total_g_loss = 0.0;

        for batch_idx in 0..num_batches {
            // ===== Train Discriminator =====
            d_optimizer.zero_grad();

            // Real images
            let start = batch_idx * batch_size * 784;
            let end = start + batch_size * 784;
            let batch_data = train_data
                .get(start..end)
                .expect("batch lies within the training set (num_batches = num_train / batch_size)")
                .to_vec();
            let real_images = RawTensor::new(batch_data, &[batch_size, 1, 28, 28], false);
            let real_labels = RawTensor::ones(&[batch_size, 1]);

            // Fake images
            let noise = randn(&[batch_size, latent_dim]);
            let fake_images = generator.forward(&noise);
            let fake_labels = RawTensor::zeros(&[batch_size, 1]);

            // D(real) should be close to 1
            let d_real = discriminator.forward(&real_images);
            let d_loss_real = bce_with_logits_loss(&d_real, &real_labels);

            // D(fake) should be close to 0
            let d_fake = discriminator.forward(&fake_images);
            let d_loss_fake = bce_with_logits_loss(&d_fake, &fake_labels);

            // Total discriminator loss
            let d_loss = d_loss_real.add(&d_loss_fake);
            total_d_loss += d_loss.item();

            d_loss.into_loss().expect("scalar loss").backward();
            d_optimizer.step();

            // ===== Train Generator =====
            g_optimizer.zero_grad();

            // Generate new fake images
            let noise = randn(&[batch_size, latent_dim]);
            let fake_images = generator.forward(&noise);

            // G wants D(fake) to be close to 1 (fool discriminator)
            let d_fake = discriminator.forward(&fake_images);
            let g_loss = bce_with_logits_loss(&d_fake, &real_labels); // Use real labels

            total_g_loss += g_loss.item();

            g_loss.into_loss().expect("scalar loss").backward();
            g_optimizer.step();
        }

        progress.update(epoch + 1);

        let avg_d_loss = total_d_loss / num_batches as f32;
        let avg_g_loss = total_g_loss / num_batches as f32;
        println!(" D_loss={avg_d_loss:.4}, G_loss={avg_g_loss:.4}");
        if !avg_d_loss.is_finite() || !avg_g_loss.is_finite() {
            eprintln!("Training diverged: D_loss={avg_d_loss}, G_loss={avg_g_loss}");
            std::process::exit(1);
        }
    }

    println!("\n=== Training Complete ===");

    let (mean_after, std_after) = mean_std(&generator.forward(&probe_noise));
    println!("Generator output after training: mean={mean_after:.4}, std={std_after:.4}");
    // A generator that collapsed (or saturated tanh) emits one constant image.
    if std_after < 1e-3 {
        eprintln!("Generator output collapsed to a constant (std={std_after})");
        std::process::exit(1);
    }

    // Generate sample images
    println!("\nGenerating sample images:");
    let noise = randn(&[1, latent_dim]);
    let generated = generator.forward(&noise);
    let gen_data = generated.data();

    // Show first 10 pixels
    println!(
        "Generated image (first 10 pixels): {:?}",
        gen_data
            .get(0..10)
            .expect("an image has at least 10 pixels")
    );
    println!("Values should be in [-1, 1] range (tanh output)");
}
