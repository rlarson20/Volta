use crate::{Result, VoltaError};
use std::fs::File;
use std::io::{ErrorKind, Read};
use std::path::Path;

/// Checks that the header's dimensions multiply to exactly the bytes left in `file`
/// after a `header_len`-byte header, and returns that byte count. Run before
/// allocating, so a corrupt header can neither overflow nor request a huge buffer.
///
/// # Errors
/// [`VoltaError::Decode`] (reported by `op`) if they don't; [`VoltaError::Io`] if the
/// file's length can't be read.
fn payload_len(
    op: &'static str,
    path: &Path,
    file: &File,
    header_len: u64,
    dims: &[(&str, u32)],
) -> Result<usize> {
    let declared = dims
        .iter()
        .try_fold(1u64, |acc, &(_, d)| acc.checked_mul(u64::from(d)));
    let remaining = file.metadata()?.len().saturating_sub(header_len);
    match declared {
        Some(n) if n == remaining => usize::try_from(n).map_err(|_| {
            VoltaError::decode(
                op,
                path,
                None,
                format!("IDX payload of {n} bytes does not fit in memory"),
            )
        }),
        _ => Err(VoltaError::decode(
            op,
            path,
            None,
            format!(
                "IDX header declares {} ({}), but the file holds {remaining} bytes after the header",
                dims.iter()
                    .map(|(name, d)| format!("{name} {d}"))
                    .collect::<Vec<_>>()
                    .join(" x "),
                declared.map_or_else(|| "overflows u64".to_string(), |n| format!("{n} bytes")),
            ),
        )),
    }
}

/// Reads the `N`-byte IDX header, or says the file is too short to hold one.
///
/// # Errors
/// [`VoltaError::Decode`] if the file ends inside the header; [`VoltaError::Io`] for any
/// other read failure.
fn read_header<const N: usize>(op: &'static str, path: &Path, file: &mut File) -> Result<[u8; N]> {
    let mut header = [0u8; N];
    file.read_exact(&mut header).map_err(|e| {
        if e.kind() == ErrorKind::UnexpectedEof {
            VoltaError::decode(
                op,
                path,
                None,
                format!("file is shorter than the {N}-byte IDX header"),
            )
        } else {
            VoltaError::Io(e)
        }
    })?;
    Ok(header)
}

/// Load MNIST images from IDX format file
/// Returns flattened image data (all images concatenated), with each pixel already
/// scaled to `[0, 1]` (`byte / 255`). Do not divide by 255 again; for `[-1, 1]` use
/// `normalize(&mut data, 0.5, 0.5)`.
/// # Errors
/// [`VoltaError::Io`] if the file can't be opened or read; [`VoltaError::Decode`],
/// carrying the path, on a short header, a wrong magic number, or a header whose
/// `count x rows x cols` is not exactly the number of pixel bytes in the file.
pub fn load_mnist_images<P: AsRef<Path>>(path: P) -> Result<Vec<f32>> {
    const OP: &str = "load_mnist_images";
    let path = path.as_ref();
    let mut file = File::open(path)?;

    // Header (16 bytes, big-endian)
    let header: [u8; 16] = read_header(OP, path, &mut file)?;
    let magic = u32::from_be_bytes([header[0], header[1], header[2], header[3]]);
    if magic != 0x0000_0803 {
        return Err(VoltaError::decode(
            OP,
            path,
            None,
            format!("invalid MNIST image magic number: 0x{magic:08x}"),
        ));
    }

    let num_images = u32::from_be_bytes([header[4], header[5], header[6], header[7]]);
    let rows = u32::from_be_bytes([header[8], header[9], header[10], header[11]]);
    let cols = u32::from_be_bytes([header[12], header[13], header[14], header[15]]);

    // Read all pixel data
    let num_pixels = payload_len(
        OP,
        path,
        &file,
        16,
        &[("images", num_images), ("rows", rows), ("cols", cols)],
    )?;
    let mut pixels = vec![0u8; num_pixels];
    file.read_exact(&mut pixels)?;

    // Convert to f32 and normalize to [0, 1]
    let data: Vec<f32> = pixels.iter().map(|&p| f32::from(p) / 255.0).collect();

    Ok(data)
}

/// Load MNIST labels from IDX format file
/// Returns label indices (0-9)
/// # Errors
/// [`VoltaError::Io`] if the file can't be opened or read; [`VoltaError::Decode`],
/// carrying the path, on a short header, a wrong magic number, or a header whose label
/// count is not exactly the number of label bytes in the file.
pub fn load_mnist_labels<P: AsRef<Path>>(path: P) -> Result<Vec<u8>> {
    const OP: &str = "load_mnist_labels";
    let path = path.as_ref();
    let mut file = File::open(path)?;

    // Header (8 bytes, big-endian)
    let header: [u8; 8] = read_header(OP, path, &mut file)?;
    let magic = u32::from_be_bytes([header[0], header[1], header[2], header[3]]);
    if magic != 0x0000_0801 {
        return Err(VoltaError::decode(
            OP,
            path,
            None,
            format!("invalid MNIST label magic number: 0x{magic:08x}"),
        ));
    }

    let num_labels = u32::from_be_bytes([header[4], header[5], header[6], header[7]]);
    let num_labels = payload_len(OP, path, &file, 8, &[("labels", num_labels)])?;

    // Read all labels
    let mut labels = vec![0u8; num_labels];
    file.read_exact(&mut labels)?;

    Ok(labels)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::index::fetch;

    /// The message of a [`VoltaError::Decode`] for `path` from loader `op`; panics on any
    /// other error, so each test also checks the variant, the path and the op (#394).
    #[expect(
        clippy::wildcard_enum_match_arm,
        reason = "test helper: every other variant is a failure"
    )]
    fn decode_message(err: VoltaError, op: &str, path: &Path) -> String {
        match err {
            VoltaError::Decode {
                path: p,
                key: None,
                detail,
            } if p == path && detail.op == op && detail.arg == Some("path") => detail.message,
            other => panic!(
                "expected a {op} Decode error for {}, got {other:?}",
                path.display()
            ),
        }
    }

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

    /// A file with the wrong magic number is a `Decode` error naming the file, not a panic
    /// or garbage pixels.
    #[test]
    fn test_load_mnist_images_rejects_bad_magic() {
        let path = write_idx("bad_magic", &[0u8; 16]);
        let err = load_mnist_images(&path).unwrap_err();
        std::fs::remove_file(&path).unwrap();
        let msg = decode_message(err, "load_mnist_images", &path);
        assert!(msg.contains("magic number: 0x00000000"), "{msg}");
    }

    /// A file that ends inside the header is a `Decode` error, not an I/O error.
    #[test]
    fn test_load_mnist_images_rejects_a_short_header() {
        let path = write_idx("short_header", &[0u8; 5]);
        let err = load_mnist_images(&path).unwrap_err();
        std::fs::remove_file(&path).unwrap();
        let msg = decode_message(err, "load_mnist_images", &path);
        assert!(msg.contains("16-byte IDX header"), "{msg}");
    }

    /// A missing file is an `Io` error.
    #[test]
    fn test_load_mnist_images_missing_file_is_io() {
        let path = std::env::temp_dir().join("volta_no_such_mnist_file");
        let err = load_mnist_images(&path).unwrap_err();
        assert!(
            matches!(&err, VoltaError::Io(e) if e.kind() == ErrorKind::NotFound),
            "{err:?}"
        );
    }

    /// An image header for `count x rows x cols`, followed by `pixels` bytes of zeros.
    fn image_file(count: u32, rows: u32, cols: u32, pixels: usize) -> Vec<u8> {
        let mut bytes = Vec::new();
        for word in [0x0000_0803, count, rows, cols] {
            bytes.extend_from_slice(&word.to_be_bytes());
        }
        bytes.resize(16 + pixels, 0);
        bytes
    }

    /// Loads `bytes` as an image file and returns its `Decode` message.
    fn image_error(name: &str, bytes: &[u8]) -> String {
        let path = write_idx(name, bytes);
        let err = load_mnist_images(&path).unwrap_err();
        std::fs::remove_file(&path).unwrap();
        decode_message(err, "load_mnist_images", &path)
    }

    /// A header whose `count x rows x cols` overflows is `InvalidData`, not an
    /// arithmetic panic (debug) or a wrapped product (release).
    #[test]
    fn test_load_mnist_images_rejects_overflowing_header() {
        let msg = image_error("overflow", &image_file(u32::MAX, u32::MAX, u32::MAX, 0));
        assert!(msg.contains("overflows"), "{msg}");
    }

    /// A header declaring far more pixels than the file holds is rejected before the
    /// buffer is allocated (it used to abort trying to allocate ~3.4 TB).
    #[test]
    fn test_load_mnist_images_rejects_header_larger_than_file() {
        let msg = image_error("too_big", &image_file(u32::MAX, 28, 28, 0));
        assert!(msg.contains("0 bytes after the header"), "{msg}");
    }

    /// Trailing bytes the header does not account for are an error, not ignored.
    #[test]
    fn test_load_mnist_images_rejects_trailing_bytes() {
        let msg = image_error("trailing", &image_file(1, 2, 2, 10));
        assert!(msg.contains("4 bytes"), "{msg}");
    }

    /// The label loader checks its count against the file the same way, in both
    /// directions, and still loads an exact file.
    #[test]
    fn test_load_mnist_labels_checks_count_against_file() {
        let labels_file = |count: u32, labels: &[u8]| {
            let mut bytes = Vec::new();
            bytes.extend_from_slice(&0x0000_0801_u32.to_be_bytes());
            bytes.extend_from_slice(&count.to_be_bytes());
            bytes.extend_from_slice(labels);
            bytes
        };
        for (name, count, labels) in [
            ("labels_short", u32::MAX, &[1u8, 2][..]),
            ("labels_long", 1, &[1u8, 2][..]),
        ] {
            let path = write_idx(name, &labels_file(count, labels));
            let err = load_mnist_labels(&path).unwrap_err();
            std::fs::remove_file(&path).unwrap();
            let msg = decode_message(err, "load_mnist_labels", &path);
            assert!(msg.contains("labels"), "{name}: {msg}");
        }
        let path = write_idx("labels_ok", &labels_file(2, &[7, 3]));
        let labels = load_mnist_labels(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert_eq!(labels, vec![7, 3]);
    }
}
