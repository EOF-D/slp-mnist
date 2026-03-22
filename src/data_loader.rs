//! Data loader for the MNIST dataset.
//! Uses `polars` to read CSV files containing the MNIST data.

use polars::prelude::*;
use std::path::Path;

pub const NUM_PIXELS: usize = 784;

/// Represents the MNIST dataset, containing samples and their corresponding labels.
pub struct Dataset {
    pub samples: Vec<f32>, // Flattened normalized pixel values. (Use a step of 784 for access.)
    pub labels: Vec<u8>,   // Unsigned 0-9.
}

impl Dataset {
    /// Load the MNIST dataset from a CSV file.
    ///
    /// # Parameters
    /// - `path`: The file path for the CSV containing MNIST data.
    ///
    /// # Returns
    /// - A `Dataset` instance containing the loaded samples and labels.
    pub fn load(path: impl AsRef<Path>) -> PolarsResult<Self> {
        let data = CsvReadOptions::default()
            .try_into_reader_with_file_path(Some(path.as_ref().to_path_buf()))?
            .finish()?;

        Ok(Self {
            samples: Self::parse_samples(&data)?,
            labels: Self::parse_labels(&data)?,
        })
    }

    /// Parse the pixel samples from the DataFrame, normalizing them to [0.0, 1.0].
    ///
    /// # Parameters
    /// - `data`: The DataFrame containing the MNIST data.
    ///
    /// # Returns
    /// - A vector of normalized pixel values for all samples, flattened.
    fn parse_samples(data: &DataFrame) -> PolarsResult<Vec<f32>> {
        let num_rows = data.height();
        let mut samples = vec![0f32; num_rows * NUM_PIXELS]; // Pre-allocate for efficiency.

        for col_idx in 1..data.width() {
            for (row_idx, val) in data[col_idx]
                .as_materialized_series()
                .i64()?
                .iter()
                .enumerate()
            {
                samples[row_idx * NUM_PIXELS + (col_idx - 1)] = val.unwrap_or(0) as f32 / 255.0;
            }
        }

        Ok(samples)
    }

    /// Parse the labels from the DataFrame.
    ///
    /// # Parameters
    /// - `data`: The DataFrame containing the MNIST data.
    ///
    /// # Returns
    /// - A vector of labels corresponding to each sample.
    ///
    /// # Errors
    /// - Returns error if the label data is missing or cannot be parsed.
    fn parse_labels(data: &DataFrame) -> PolarsResult<Vec<u8>> {
        data[0]
            .as_materialized_series()
            .i64()?
            .iter()
            .map(|val| {
                val.map(|v| v as u8)
                    .ok_or_else(|| PolarsError::NoData("no label data".into()))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_PATH: &str = "datasets/mnist_test.csv";
    const TRAIN_PATH: &str = "datasets/mnist_train.csv";

    #[test]
    fn test_test_dataset_size() {
        let dataset = Dataset::load(TEST_PATH).unwrap();

        assert_eq!(dataset.labels.len(), 10000);
        assert_eq!(dataset.samples.len(), 10000 * NUM_PIXELS);
    }

    #[test]
    fn test_train_dataset_size() {
        let dataset = Dataset::load(TRAIN_PATH).unwrap();

        assert_eq!(dataset.labels.len(), 60000);
        assert_eq!(dataset.samples.len(), 60000 * NUM_PIXELS);
    }

    #[test]
    fn test_normalized() {
        let dataset = Dataset::load(TRAIN_PATH).unwrap();
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
        let dataset = Dataset::load(TRAIN_PATH).unwrap();

        assert!(
            dataset.labels.iter().all(|&l| l <= 9),
            "label out of 0-9 range"
        );
    }
}
