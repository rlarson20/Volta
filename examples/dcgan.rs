use volta::{
    Conv2d, ConvTranspose2d, Flatten, Linear, Module, ProgressBar, RawTensor, SGD, Tanh, TensorOps,
    bce_with_logits_loss, load_mnist_images, normalize, randn,
};

// Generator: noise -> fake images
struct Generator {
    fc: Linear,
    deconv1: ConvTranspose2d,
    deconv2: ConvTranspose2d,
    tanh: Tanh,
}

impl Generator {
    fn new(latent_dim: usize) -> Self {
        Generator {
            fc: Linear::new(latent_dim, 256 * 7 * 7, true),
            deconv1: ConvTranspose2d::new(256, 128, 4, 2, 1, true),
            deconv2: ConvTranspose2d::new(128, 1, 4, 2, 1, true),
            tanh: Tanh,
        }
    }

    fn forward(&self, z: &volta::Tensor) -> volta::Tensor {
        let batch = z.borrow().shape[0];

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

    fn parameters(&self) -> Vec<volta::Tensor> {
        let mut params = Vec::new();
        params.extend(self.fc.parameters());
        params.extend(self.deconv1.parameters());
        params.extend(self.deconv2.parameters());
        params
    }
}

// Discriminator: images -> real/fake classification
struct Discriminator {
    conv1: Conv2d,
    conv2: Conv2d,
    flatten: Flatten,
    fc: Linear,
}

impl Discriminator {
    fn new() -> Self {
        Discriminator {
            conv1: Conv2d::new(1, 64, 4, 2, 1, true),
            conv2: Conv2d::new(64, 128, 4, 2, 1, true),
            flatten: Flatten,
            fc: Linear::new(128 * 7 * 7, 1, true),
        }
    }

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

    fn parameters(&self) -> Vec<volta::Tensor> {
        let mut params = Vec::new();
        params.extend(self.conv1.parameters());
        params.extend(self.conv2.parameters());
        params.extend(self.fc.parameters());
        params
    }
}

fn main() {
    println!("=== DCGAN on MNIST ===\n");

    let latent_dim = 100;
    let batch_size = 64;
    let num_epochs = 5;
    let learning_rate = 0.0002;

    // Load MNIST data
    let (train_images, num_train) = match load_mnist_images("data/mnist/train-images-idx3-ubyte") {
        Ok(images) => {
            let n = images.len() / 784;
            println!("Loaded {} MNIST training images", n);
            (images, n)
        }
        Err(_) => {
            println!("Could not load MNIST data from data/mnist/ directory");
            println!("Generating synthetic data for demonstration...\n");
            let n = 1000;
            let images = vec![0.5; n * 784];
            (images, n)
        }
    };

    // Normalize to [-1, 1] to match tanh output
    let mut train_data = train_images;
    normalize(&mut train_data, 127.5, 127.5);

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

    let mut g_optimizer = SGD::new(generator.parameters(), learning_rate, 0.5, 0.0);
    let mut d_optimizer = SGD::new(discriminator.parameters(), learning_rate, 0.5, 0.0);

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
            let batch_data = train_data[start..end].to_vec();
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
            total_d_loss += d_loss.borrow().data[0];

            d_loss.backward();
            d_optimizer.step();

            // ===== Train Generator =====
            g_optimizer.zero_grad();

            // Generate new fake images
            let noise = randn(&[batch_size, latent_dim]);
            let fake_images = generator.forward(&noise);

            // G wants D(fake) to be close to 1 (fool discriminator)
            let d_fake = discriminator.forward(&fake_images);
            let g_loss = bce_with_logits_loss(&d_fake, &real_labels); // Use real labels

            total_g_loss += g_loss.borrow().data[0];

            g_loss.backward();
            g_optimizer.step();
        }

        progress.update(epoch + 1);

        let avg_d_loss = total_d_loss / num_batches as f32;
        let avg_g_loss = total_g_loss / num_batches as f32;
        println!(" D_loss={:.4}, G_loss={:.4}", avg_d_loss, avg_g_loss);
    }

    println!("\n=== Training Complete ===");

    // Generate sample images
    println!("\nGenerating sample images:");
    let noise = randn(&[1, latent_dim]);
    let generated = generator.forward(&noise);
    let gen_data = &generated.borrow().data;

    // Show first 10 pixels
    println!("Generated image (first 10 pixels): {:?}", &gen_data[0..10]);
    println!("Values should be in [-1, 1] range (tanh output)");
}
