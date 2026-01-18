use std::path::Path;
use volta::{
    Conv2d, DataLoader, Dropout, Flatten, Linear, MaxPool2d, Module, ProgressBar, RawTensor, ReLU,
    SGD, Sequential, TensorOps, load_mnist_images, load_mnist_labels, to_one_hot,
};

fn main() {
    println!("=== MNIST CNN Training ===\n");

    // Try to load MNIST data
    let mnist_dir = "data/mnist";
    let train_images_path = format!("{}/train-images-idx3-ubyte", mnist_dir);
    let train_labels_path = format!("{}/train-labels-idx1-ubyte", mnist_dir);

    let (image_data, label_data, num_samples) = if Path::new(&train_images_path).exists()
        && Path::new(&train_labels_path).exists()
    {
        println!("Loading MNIST data from {}...", mnist_dir);

        let images = load_mnist_images(&train_images_path).expect("Failed to load MNIST images");
        let labels = load_mnist_labels(&train_labels_path).expect("Failed to load MNIST labels");

        let num_samples = labels.len();
        let label_data = to_one_hot(&labels, 10);

        println!("Loaded {} training samples", num_samples);
        (images, label_data, num_samples)
    } else {
        println!("MNIST data not found. Using synthetic data for demonstration.");
        println!("\nTo download MNIST data:");
        println!("  mkdir -p data/mnist");
        println!("  cd data/mnist");
        println!("  curl -O http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz");
        println!("  curl -O http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz");
        println!("  gunzip *.gz");
        println!();

        // Generate synthetic data
        let num_samples = 1024;
        let image_data: Vec<f32> = (0..num_samples * 28 * 28)
            .map(|i| (i % 255) as f32 / 255.0)
            .collect();
        let labels: Vec<u8> = (0..num_samples).map(|i| (i % 10) as u8).collect();
        let label_data = to_one_hot(&labels, 10);

        println!("Generated {} synthetic samples", num_samples);
        (image_data, label_data, num_samples)
    };

    // Create DataLoader
    let batch_size = 64;
    let mut dataloader = DataLoader::new(
        image_data,
        label_data,
        &[1, 28, 28], // NCHW format: 1 channel, 28x28 pixels
        &[10],        // 10-class one-hot labels
        batch_size,
        true, // shuffle
    );

    // Build CNN matching PyTorch MNIST example
    // Input shape: [B, 1, 28, 28]
    // After Conv2d(1,32,3,s=1,p=0): [B, 32, 26, 26]
    // After Conv2d(32,64,3,s=1,p=0): [B, 64, 24, 24]
    // After MaxPool2d(k=2,s=2): [B, 64, 12, 12]
    // After Flatten: [B, 9216]
    let mut model = Sequential::new(vec![
        Box::new(Conv2d::new(1, 32, 3, 1, 0, true)),
        Box::new(ReLU),
        Box::new(Conv2d::new(32, 64, 3, 1, 0, true)),
        Box::new(ReLU),
        Box::new(MaxPool2d::new(2, 2, 0)),
        Box::new(Dropout::new(0.25)),
        Box::new(Flatten),
        Box::new(Linear::new(9216, 128, true)),
        Box::new(ReLU),
        Box::new(Dropout::new(0.5)),
        Box::new(Linear::new(128, 10, true)),
    ]);

    let mut optimizer = SGD::new(model.parameters(), 0.01, 0.9, 0.0);

    println!("\nArchitecture:");
    println!("  Conv2d(1→32, k=3) → ReLU → Conv2d(32→64, k=3) → ReLU");
    println!("  MaxPool(2×2) → Dropout(0.25) → Flatten");
    println!("  Linear(9216→128) → ReLU → Dropout(0.5) → Linear(128→10)");
    println!(
        "\nTraining with {} batches per epoch...\n",
        num_samples / batch_size
    );

    // Training
    model.train(true);
    let num_epochs = 10;
    let total_batches = num_samples / batch_size;

    for epoch in 0..num_epochs {
        dataloader.reset();
        let mut epoch_loss = 0.0;
        let mut num_batches = 0;

        let mut progress = ProgressBar::new(
            total_batches,
            &format!("Epoch {:2}/{}", epoch + 1, num_epochs),
        );

        for (batch_x, batch_y) in &mut dataloader {
            optimizer.zero_grad();
            let output = model.forward(&batch_x);
            let loss = RawTensor::cross_entropy_loss(&output, &batch_y);
            loss.backward();
            optimizer.step();

            epoch_loss += loss.borrow().data.first().copied().unwrap_or(0.0);
            num_batches += 1;
            progress.update(num_batches);
        }

        let avg_loss = epoch_loss / num_batches as f32;
        println!(" Loss = {:.6}", avg_loss);
    }

    // Test accuracy
    println!("\n=== Testing ===");
    model.train(false);
    dataloader.reset();

    let mut total_correct = 0;
    let mut total_samples = 0;
    let mut progress = ProgressBar::new(total_batches, "Testing");

    for (test_x, test_y) in &mut dataloader {
        let output = model.forward(&test_x);
        let pred_data = &output.borrow().data;
        let target_data = &test_y.borrow().data;
        let batch_size = test_y.borrow().shape.first().copied().unwrap_or(1);

        for i in 0..batch_size {
            let pred_class = (0..10)
                .max_by(|&a, &b| {
                    let val_a = pred_data.get(i * 10 + a).copied().unwrap_or(f32::NAN);
                    let val_b = pred_data.get(i * 10 + b).copied().unwrap_or(f32::NAN);
                    val_a.partial_cmp(&val_b).unwrap()
                })
                .unwrap();

            let true_class = (0..10)
                .position(|j| target_data.get(i * 10 + j).copied().unwrap_or(0.0) > 0.5)
                .unwrap();

            if pred_class == true_class {
                total_correct += 1;
            }
            total_samples += 1;
        }
        progress.inc();
    }
    println!();

    let accuracy = total_correct as f32 / total_samples as f32 * 100.0;
    println!(
        "Training Set Accuracy: {}/{} ({:.2}%)",
        total_correct, total_samples, accuracy
    );
}
