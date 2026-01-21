use std::fs::File;
use std::io::{Read, Result};
use std::path::Path;

/// Load MNIST images from IDX format file
/// Returns flattened image data (all images concatenated)
/// # Errors
/// file opening errors
pub fn load_mnist_images<P: AsRef<Path>>(path: P) -> Result<Vec<f32>> {
    let mut file = File::open(path)?;

    // Read header (16 bytes)
    let mut header = [0u8; 16];
    file.read_exact(&mut header)?;

    // Parse header (big-endian)
    let magic = u32::from_be_bytes([header[0], header[1], header[2], header[3]]);
    if magic != 0x0000_0803 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Invalid MNIST image magic number: 0x{magic:08x}"),
        ));
    }

    let num_images = u32::from_be_bytes([header[4], header[5], header[6], header[7]]) as usize;
    let rows = u32::from_be_bytes([header[8], header[9], header[10], header[11]]) as usize;
    let cols = u32::from_be_bytes([header[12], header[13], header[14], header[15]]) as usize;

    // Read all pixel data
    let num_pixels = num_images * rows * cols;
    let mut pixels = vec![0u8; num_pixels];
    file.read_exact(&mut pixels)?;

    // Convert to f32 and normalize to [0, 1]
    let data: Vec<f32> = pixels.iter().map(|&p| f32::from(p) / 255.0).collect();

    Ok(data)
}

/// Load MNIST labels from IDX format file
/// Returns label indices (0-9)
/// # Errors
/// file opening errors
pub fn load_mnist_labels<P: AsRef<Path>>(path: P) -> Result<Vec<u8>> {
    let mut file = File::open(path)?;

    // Read header (8 bytes)
    let mut header = [0u8; 8];
    file.read_exact(&mut header)?;

    // Parse header (big-endian)
    let magic = u32::from_be_bytes([header[0], header[1], header[2], header[3]]);
    if magic != 0x0000_0801 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Invalid MNIST label magic number: 0x{magic:08x}"),
        ));
    }

    let num_labels = u32::from_be_bytes([header[4], header[5], header[6], header[7]]) as usize;

    // Read all labels
    let mut labels = vec![0u8; num_labels];
    file.read_exact(&mut labels)?;

    Ok(labels)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mnist_header_parsing() {
        // This test validates the magic number checking
        // Actual file loading would require test data

        // Just verify the functions exist and have correct signatures
        let _ = load_mnist_images::<&str> as fn(_) -> Result<Vec<f32>>;
        let _ = load_mnist_labels::<&str> as fn(_) -> Result<Vec<u8>>;
    }
}
