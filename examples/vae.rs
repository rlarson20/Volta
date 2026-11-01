//! Variational autoencoder on MNIST: a `Linear` encoder producing `mu` and `logvar`, reparameterized sampling, and a `Linear` decoder, trained with `SGD` on reconstruction (MSE) plus KL divergence.
//!
//! Needs the MNIST training images in `data/mnist/`; it exits with download instructions if they are missing, rather than train on made-up data.
use volta::{
    Linear, Module, ProgressBar, RawTensor, SGD, kl_divergence_gaussian, load_mnist_images,
    mse_loss, randn_like,
};

fn main() {
    println!("=== Variational Autoencoder (VAE) on MNIST ===\n");

    // Hyperparameters
    let latent_dim = 20;
    let hidden_dim = 400;
    let input_dim = 784; // 28x28 flattened
    let batch_size = 100;
    let num_epochs = 10;
    let learning_rate = 0.001;

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
    let num_train = train_images.len() / input_dim;
    println!("Loaded {num_train} MNIST training images");

    // Normalize to [0, 1]
    // `load_mnist_images` already scales pixels to [0, 1].
    let train_data = train_images;

    println!("Architecture:");
    println!("  Encoder: {input_dim} -> {hidden_dim} -> {latent_dim} (mu + logvar)");
    println!("  Decoder: {latent_dim} -> {hidden_dim} -> {input_dim}");
    println!("  Latent dimension: {latent_dim}");
    println!("  Batch size: {batch_size}");
    println!("\nStarting training...\n");

    // Build encoder: input -> hidden -> (mu, logvar)
    let enc_fc1 = Linear::new(input_dim, hidden_dim, true);
    let enc_mu = Linear::new(hidden_dim, latent_dim, true);
    let enc_logvar = Linear::new(hidden_dim, latent_dim, true);

    // Build decoder: latent -> hidden -> output
    let dec_fc1 = Linear::new(latent_dim, hidden_dim, true);
    let dec_out = Linear::new(hidden_dim, input_dim, true);

    // Collect all parameters for optimizer
    let mut params = Vec::new();
    params.extend(enc_fc1.parameters());
    params.extend(enc_mu.parameters());
    params.extend(enc_logvar.parameters());
    params.extend(dec_fc1.parameters());
    params.extend(dec_out.parameters());

    let mut optimizer = SGD::new(params, learning_rate, 0.9, 0.0);

    let num_batches = num_train / batch_size;
    let mut progress = ProgressBar::new(num_epochs, "Training");

    // Both terms are per-sample ELBO terms: `mse_loss` is a mean over every pixel of the
    // batch, so scale it back up by `input_dim` to a per-image sum; the KL is a sum over
    // the batch and latent dims, so divide it by the batch size.
    let recon_scale = RawTensor::new(vec![input_dim as f32], &[1], false);
    let kl_scale = RawTensor::new(vec![1.0 / batch_size as f32], &[1], false);

    let mut first_epoch_loss = None;
    let mut last_epoch_loss = f32::NAN;

    for epoch in 0..num_epochs {
        let mut total_loss = 0.0;
        let mut total_recon = 0.0;
        let mut total_kl = 0.0;

        for batch_idx in 0..num_batches {
            let start = batch_idx * batch_size * input_dim;
            let end = start + batch_size * input_dim;
            let batch_data = train_data
                .get(start..end)
                .expect("the batch is inside the training data")
                .to_vec();
            let x = RawTensor::new(batch_data.clone(), &[batch_size, input_dim], false);

            // Encoder forward
            let h = enc_fc1.forward(&x).relu();
            let mu = enc_mu.forward(&h);
            let logvar = enc_logvar.forward(&h);

            // Reparameterization: z = mu + sigma * epsilon
            let epsilon = randn_like(&mu);
            let std = logvar
                .elem_mul(&RawTensor::new(vec![0.5], &[1], false))
                .exp();
            let z = mu.add(&std.elem_mul(&epsilon));

            // Decoder forward
            let h_dec = dec_fc1.forward(&z).relu();
            let x_recon = dec_out.forward(&h_dec).sigmoid(); // Output in [0, 1]

            // Loss = Reconstruction + KL divergence
            let recon_loss = mse_loss(&x_recon, &x).elem_mul(&recon_scale);
            let kl_loss = kl_divergence_gaussian(&mu, &logvar).elem_mul(&kl_scale);
            let loss = recon_loss.add(&kl_loss);

            total_loss += loss.item();
            total_recon += recon_loss.item();
            total_kl += kl_loss.item();

            optimizer.zero_grad();
            loss.into_loss().expect("scalar loss").backward();
            optimizer.step();
        }

        progress.update(epoch + 1);

        let avg_loss = total_loss / num_batches as f32;
        let avg_recon = total_recon / num_batches as f32;
        let avg_kl = total_kl / num_batches as f32;
        if !avg_loss.is_finite() {
            eprintln!(
                "\nEpoch {}: loss is {avg_loss} (Recon={avg_recon}, KL={avg_kl}); training diverged",
                epoch + 1
            );
            std::process::exit(1);
        }
        first_epoch_loss.get_or_insert(avg_loss);
        last_epoch_loss = avg_loss;
        if epoch % 2 == 1 || epoch == num_epochs - 1 {
            println!(" Loss={avg_loss:.4} (Recon={avg_recon:.4}, KL={avg_kl:.4})");
        }
    }

    let first_epoch_loss = first_epoch_loss.expect("at least one epoch ran");
    if last_epoch_loss >= first_epoch_loss {
        eprintln!(
            "\nThe loss did not fall: {first_epoch_loss:.4} in epoch 1, {last_epoch_loss:.4} in the last epoch"
        );
        std::process::exit(1);
    }

    println!("\n=== Training Complete ===");

    // Test reconstruction
    println!("\nTesting reconstruction on first image:");
    let test_img = train_data
        .get(0..input_dim)
        .expect("the training data holds at least one image")
        .to_vec();
    let x_test = RawTensor::new(test_img.clone(), &[1, input_dim], false);

    let h = enc_fc1.forward(&x_test).relu();
    let mu = enc_mu.forward(&h);
    let logvar = enc_logvar.forward(&h);

    let epsilon = randn_like(&mu);
    let std = logvar
        .elem_mul(&RawTensor::new(vec![0.5], &[1], false))
        .exp();
    let z = mu.add(&std.elem_mul(&epsilon));

    let h_dec = dec_fc1.forward(&z).relu();
    let x_recon = dec_out.forward(&h_dec).sigmoid();

    let recon_data = x_recon.data();
    let orig_data = x_test.data();

    // Show first 10 pixels
    println!(
        "Original (first 10 pixels): {:?}",
        orig_data
            .get(0..10)
            .expect("an image has at least 10 pixels")
    );
    println!(
        "Reconstructed (first 10):    {:?}",
        recon_data
            .get(0..10)
            .expect("a reconstruction has at least 10 pixels")
    );

    let mse = orig_data
        .iter()
        .zip(recon_data.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        / input_dim as f32;
    println!("Reconstruction MSE: {mse:.6}");
    if !mse.is_finite() {
        eprintln!("The test reconstruction is not finite");
        std::process::exit(1);
    }
}
