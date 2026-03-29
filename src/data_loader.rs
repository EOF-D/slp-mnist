//! Data loader for the MNIST dataset.
//! IDX file format: <https://www.fon.hum.uva.nl/praat/manual/IDX_file_format.html>

use std::{
    fs::File,
    io::{self, BufReader, Read},
    path::Path,
};

use rand::prelude::SliceRandom;

pub const NUM_PIXELS: usize = 784;

// `u32` hex literals as stated in the IDX format documentation.
const MAGIC_IMAGES: u32 = 0x0000_0803; // unsigned byte, count, rows, cols.
const MAGIC_LABELS: u32 = 0x0000_0801; // unsigned byte, count.

/// Represents the MNIST dataset, containing samples and their corresponding labels.
pub struct Dataset {
    pub samples: Vec<f32>, // Flattened normalized pixel values. (Use a step of 784 for access.)
    pub labels: Vec<u8>,   // Unsigned 0-9.
}

impl Dataset {
    /// Load the MNIST dataset from a pair of IDX ubyte files.
    ///
    /// # Parameters
    /// - `images_path`: The file path for the IDX3 ubyte file containing MNIST images.
    /// - `labels_path`: The file path for the IDX1 ubyte file containing MNIST labels.
    ///
    /// # Returns
    /// - A `Dataset` instance containing the shuffled samples and labels.
    ///
    /// # References
    ///  - <https://doc.rust-lang.org/std/convert/trait.AsRef.html>
    ///  - <https://doc.rust-lang.org/std/path/struct.Path.html>
    pub fn load(images_path: impl AsRef<Path>, labels_path: impl AsRef<Path>) -> io::Result<Self> {
        let mut samples = Self::parse_samples(images_path)?;
        let mut labels = Self::parse_labels(labels_path)?;

        // Shuffle the dataset for random ordering when training.
        Self::shuffle(&mut samples, &mut labels);
        Ok(Self { samples, labels })
    }

    /// Get the number of samples in the dataset.
    ///
    /// # Returns
    /// - The number of samples.
    pub fn len(&self) -> usize {
        self.labels.len()
    }

    /// Get the pixel slice for the sample at the given index.
    ///
    /// # Parameters
    /// - `i`: The index of the sample to get.
    ///
    /// # Returns
    /// - A slice of normalized pixel values for the sample at the index.
    pub fn sample(&self, i: usize) -> &[f32] {
        &self.samples[i * NUM_PIXELS..(i + 1) * NUM_PIXELS]
    }

    /// Parse the pixel samples from an IDX3 ubyte file, normalizing them to [0.0, 1.0].
    ///
    /// # Parameters
    /// - `path`: The file path for the IDX3 ubyte file containing MNIST images.
    ///
    /// # Returns
    /// - A vector of normalized pixel values for all samples, flattened.
    fn parse_samples(path: impl AsRef<Path>) -> io::Result<Vec<f32>> {
        let mut reader = BufReader::new(File::open(path)?);
        guard_magic(&mut reader, MAGIC_IMAGES)?;

        // Header encodes as big-endian u32.
        // Cast to `usize` for length & indexing.
        let count = read_u32_be(&mut reader)? as usize;
        let rows = read_u32_be(&mut reader)? as usize;
        let columns = read_u32_be(&mut reader)? as usize;

        // Pre-allocate a vector for all pixel data, then read into.
        let mut pixels = vec![0u8; count * rows * columns];
        reader.read_exact(&mut pixels)?;

        Ok(pixels.iter().map(|&v| v as f32 / 255.0).collect())
    }

    /// Parse the labels from an IDX1 ubyte file.
    ///
    /// # Parameters
    /// - `path`: The file path for the IDX1 ubyte file containing MNIST labels.
    ///
    /// # Returns
    /// - A vector of labels corresponding to each sample.
    fn parse_labels(path: impl AsRef<Path>) -> io::Result<Vec<u8>> {
        let mut reader = BufReader::new(File::open(path)?);
        guard_magic(&mut reader, MAGIC_LABELS)?;

        let count = read_u32_be(&mut reader)? as usize;

        // Pre-allocate a vector for all labels, then read into.
        let mut labels = vec![0u8; count];
        reader.read_exact(&mut labels)?;

        Ok(labels)
    }

    /// Shuffle samples and labels together.
    ///
    /// # Parameters
    /// - `samples`: The flattened pixel vector to shuffle.
    /// - `labels`: The label vector to shuffle with `samples`.
    fn shuffle(samples: &mut Vec<f32>, labels: &mut Vec<u8>) {
        let num = labels.len();
        let mut rng = rand::rng();

        // Create a randomized vector of indices, then reorder according to it.
        let mut indices: Vec<usize> = (0..num).collect();
        indices.shuffle(&mut rng);

        let shuffled_samples: Vec<f32> = indices
            .iter()
            .flat_map(|&i| {
                let start = i * NUM_PIXELS;
                let end = start + NUM_PIXELS;
                samples[start..end].iter().copied()
            })
            .collect();

        let shuffled_labels: Vec<u8> = indices.iter().map(|&i| labels[i]).collect();

        // Update the original vectors with the shuffled data.
        *samples = shuffled_samples;
        *labels = shuffled_labels;
    }
}

fn guard_magic(reader: &mut impl Read, expected: u32) -> io::Result<()> {
    // Read magic number as big-endian u32 and compare to expected value for images or labels.
    let magic = read_u32_be(reader)?;
    if magic != expected {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("bad magic number: expected {expected:#010x}, got {magic:#010x}"),
        ));
    }

    Ok(())
}

fn read_u32_be(reader: &mut impl Read) -> io::Result<u32> {
    // Make a buffer of 4 bytes to read the u32 value into.
    let mut buffer = [0u8; 4];
    reader.read_exact(&mut buffer)?;

    // Re-interpret as u32 and return.
    Ok(u32::from_be_bytes(buffer))
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_IMAGES: &str = "datasets/t10k-images.idx3-ubyte";
    const TEST_LABELS: &str = "datasets/t10k-labels.idx1-ubyte";
    const TRAIN_IMAGES: &str = "datasets/train-images.idx3-ubyte";
    const TRAIN_LABELS: &str = "datasets/train-labels.idx1-ubyte";

    #[test]
    fn test_test_dataset_size() {
        let dataset = Dataset::load(TEST_IMAGES, TEST_LABELS).unwrap();

        assert_eq!(dataset.labels.len(), 10000);
        assert_eq!(dataset.samples.len(), 10000 * NUM_PIXELS);
    }

    #[test]
    fn test_train_dataset_size() {
        let dataset = Dataset::load(TRAIN_IMAGES, TRAIN_LABELS).unwrap();

        assert_eq!(dataset.labels.len(), 60000);
        assert_eq!(dataset.samples.len(), 60000 * NUM_PIXELS);
    }

    #[test]
    fn test_normalized() {
        let dataset = Dataset::load(TRAIN_IMAGES, TRAIN_LABELS).unwrap();
        let min = dataset
            .samples
            .iter()
            .cloned()
            .fold(f32::INFINITY, f32::min);

        let max = dataset
            .samples
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);

        assert!(min >= 0.0, "min pixel value below 0: {min}");
        assert!(max <= 1.0, "max pixel value above 1: {max}");
    }

    #[test]
    fn test_labels_valid() {
        let dataset = Dataset::load(TRAIN_IMAGES, TRAIN_LABELS).unwrap();

        assert!(
            dataset.labels.iter().all(|&l| l <= 9),
            "label out of 0-9 range"
        );
    }
}
