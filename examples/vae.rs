use volta::{
    Linear, Module, ProgressBar, RawTensor, SGD, TensorOps, kl_divergence_gaussian,
    load_mnist_images, mse_loss, normalize, randn_like,
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
    let (train_images, num_train) = match load_mnist_images("data/train-images-idx3-ubyte") {
        Ok(images) => {
            let n = images.len() / input_dim;
            println!("Loaded {} MNIST training images", n);
            (images, n)
        }
        Err(_) => {
            println!("Could not load MNIST data from data/ directory");
            println!("Generating synthetic data for demonstration...\n");
            let n = 1000;
            let images = vec![0.5; n * input_dim];
            (images, n)
        }
    };

    // Normalize to [0, 1]
    let mut train_data = train_images;
    normalize(&mut train_data, 0.0, 255.0);

    println!("Architecture:");
    println!(
        "  Encoder: {} -> {} -> {} (mu + logvar)",
        input_dim, hidden_dim, latent_dim
    );
    println!(
        "  Decoder: {} -> {} -> {}",
        latent_dim, hidden_dim, input_dim
    );
    println!("  Latent dimension: {}", latent_dim);
    println!("  Batch size: {}", batch_size);
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

    for epoch in 0..num_epochs {
        optimizer.zero_grad();
        let mut total_loss = 0.0;
        let mut total_recon = 0.0;
        let mut total_kl = 0.0;

        for batch_idx in 0..num_batches {
            let start = batch_idx * batch_size * input_dim;
            let end = start + batch_size * input_dim;
            let batch_data = train_data.get(start..end).unwrap_or(&[]).to_vec();
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
            let recon_loss = mse_loss(&x_recon, &x);
            let kl_loss = kl_divergence_gaussian(&mu, &logvar);
            let loss = recon_loss.add(&kl_loss);

            total_loss += loss.borrow().data.first().copied().unwrap_or(f32::NAN);
            total_recon += recon_loss
                .borrow()
                .data
                .first()
                .copied()
                .unwrap_or(f32::NAN);
            total_kl += kl_loss.borrow().data.first().copied().unwrap_or(f32::NAN);

            loss.backward();
        }

        optimizer.step();
        progress.update(epoch + 1);

        if epoch % 2 == 1 || epoch == num_epochs - 1 {
            let avg_loss = total_loss / num_batches as f32;
            let avg_recon = total_recon / num_batches as f32;
            let avg_kl = total_kl / num_batches as f32;
            println!(
                " Loss={:.4} (Recon={:.4}, KL={:.4})",
                avg_loss, avg_recon, avg_kl
            );
        }
    }

    println!("\n=== Training Complete ===");

    // Test reconstruction
    println!("\nTesting reconstruction on first image:");
    let test_img = train_data.get(0..input_dim).unwrap_or(&[]).to_vec();
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

    let recon_data = &x_recon.borrow().data;
    let orig_data = &x_test.borrow().data;

    // Show first 10 pixels
    println!(
        "Original (first 10 pixels): {:?}",
        &orig_data.get(0..10).unwrap_or(&[])
    );
    println!(
        "Reconstructed (first 10):    {:?}",
        &recon_data.get(0..10).unwrap_or(&[])
    );

    let mse = orig_data
        .iter()
        .zip(recon_data.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        / input_dim as f32;
    println!("Reconstruction MSE: {:.6}", mse);
}
