use std::fs::File;
use std::io::{Read, Result};
use std::path::Path;

/// Load MNIST images from IDX format file
/// Returns flattened image data (all images concatenated), with each pixel already
/// scaled to `[0, 1]` (`byte / 255`). Do not divide by 255 again; for `[-1, 1]` use
/// `normalize(&mut data, 0.5, 0.5)`.
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

    use crate::index::fetch;

    /// Writes `bytes` to a uniquely named file in the temp dir and returns its path.
    fn write_idx(name: &str, bytes: &[u8]) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!("volta_{}_{name}", std::process::id()));
        std::fs::write(&path, bytes).expect("write temp IDX file");
        path
    }

    /// Pixels come back as `byte / 255`, so the extremes are exactly 0 and 1.
    #[test]
    fn test_load_mnist_images_scales_pixels_to_unit_range() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&0x0000_0803_u32.to_be_bytes());
        bytes.extend_from_slice(&1_u32.to_be_bytes()); // images
        bytes.extend_from_slice(&1_u32.to_be_bytes()); // rows
        bytes.extend_from_slice(&3_u32.to_be_bytes()); // cols
        bytes.extend_from_slice(&[0, 128, 255]);
        let path = write_idx("images", &bytes);

        let data = load_mnist_images(&path).unwrap();
        std::fs::remove_file(&path).unwrap();

        assert_eq!(data.len(), 3);
        assert_eq!(fetch(&data, 0), 0.0);
        assert!((fetch(&data, 1) - 128.0 / 255.0).abs() < 1e-7);
        assert_eq!(fetch(&data, 2), 1.0);
    }

    /// A file with the wrong magic number is `InvalidData`, not a panic or garbage pixels.
    #[test]
    fn test_load_mnist_images_rejects_bad_magic() {
        let path = write_idx("bad_magic", &[0u8; 16]);
        let err = load_mnist_images(&path).unwrap_err();
        std::fs::remove_file(&path).unwrap();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    }
}
