//! Parallel MNIST classifier using mini batching gradient accumulation, and batched inference.

use crate::data::{NUM_CLASSES, NUM_PIXELS};
use crate::math::{compute_gradients, gradient_step};
use crate::sequential::SequentialModel;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Parallelizes an SLP model.
#[derive(Serialize, Deserialize)]
pub struct ParallelModel {
    #[serde(flatten)]
    pub inner: SequentialModel,

    batch_size: usize,
}

impl ParallelModel {
    /// Create a new model with random weights.
    ///
    /// # Parameters
    /// - `batch_size`: The number of samples per gradient update.
    ///
    /// # Returns
    /// - A `ParallelModel` instance with weights randomized from [-0.01, 0.01].
    pub fn new(batch_size: usize) -> Self {
        Self {
            inner: SequentialModel::new(),
            batch_size,
        }
    }

    /// Train for an epoch over the whole dataset in parallel.
    ///
    /// # Parameters
    /// - `samples`: Flattened pixel values for the whole dataset.
    /// - `labels`: The correct label for each sample in the dataset.
    /// - `learning_rate`: The learning rate value for the gradient descent step.
    pub fn train_epoch(&mut self, samples: &[f32], labels: &[u8], learning_rate: f32) {
        let batches = samples
            .chunks(self.batch_size * NUM_PIXELS)
            .zip(labels.chunks(self.batch_size));

        for (batch_samples, batch_labels) in batches {
            self.batch_train(batch_samples, batch_labels, learning_rate);
        }
    }

    /// Compute the accuracy of the model over a batch of samples in parallel.
    ///
    /// # Parameters
    /// - `samples`: Flattened pixel values.
    /// - `labels`: The correct label for each sample.
    ///
    /// # Returns
    /// - The accuracy as a percentage.
    pub fn batch_test(&self, samples: &[f32], labels: &[u8]) -> f32 {
        let correct = samples
            .par_chunks_exact(NUM_PIXELS)
            .zip(labels.par_iter())
            .filter(|(pixels, label)| self.inner.predict(pixels) == **label)
            .count();

        correct as f32 / labels.len() as f32 * 100.0
    }

    /// Train over a single batch in parallel.
    ///
    /// # Parameters
    /// - `samples`: Flattened pixel values for the batch.
    /// - `labels`: The correct label for each sample in the batch.
    /// - `learning_rate`: The learning rate value for the gradient descent step.
    fn batch_train(&mut self, samples: &[f32], labels: &[u8], learning_rate: f32) {
        // Compute gradients for each sample in parallel.
        let gradients: Vec<(Vec<f32>, Vec<f32>)> = samples
            .par_chunks_exact(NUM_PIXELS)
            .zip(labels.par_iter())
            .map(|(pixels, &label)| {
                let probabilities = self.inner.forward_pass(pixels);
                compute_gradients(pixels, &probabilities, label)
            })
            .collect();

        // Accumulate gradients from samples into buffer.
        let mut weight_gradients = vec![0.0f32; NUM_CLASSES * NUM_PIXELS];
        let mut bias_gradients = vec![0.0f32; NUM_CLASSES];

        for (wg, bg) in gradients {
            weight_gradients
                .iter_mut()
                .zip(wg)
                .for_each(|(x, y)| *x += y);

            bias_gradients.iter_mut().zip(bg).for_each(|(x, y)| *x += y);
        }

        // Divide learning rate by batch size to get average gradient across the batch size.
        let lr = learning_rate / labels.len() as f32;

        for c in 0..NUM_CLASSES {
            let start = c * NUM_PIXELS;

            gradient_step(
                &mut self.inner.weights[start..start + NUM_PIXELS],
                &weight_gradients[start..start + NUM_PIXELS],
                lr,
            );

            self.inner.biases[c] -= lr * bias_gradients[c];
        }
    }
}

impl Default for ParallelModel {
    fn default() -> Self {
        Self::new(32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_batch_test() {
        let model = ParallelModel::default();
        let samples = vec![0.1; 32 * NUM_PIXELS];
        let labels = vec![0; 32];

        let accuracy = model.batch_test(&samples, &labels);

        assert!(accuracy >= 0.0 && accuracy <= 100.0);
    }

    #[test]
    fn test_model_train_epoch_updates_weights() {
        let mut model = ParallelModel::default();
        let before = model.inner.weights.clone();
        let samples = vec![0.1; 32 * NUM_PIXELS];
        let labels = vec![1; 32];

        model.train_epoch(&samples, &labels, 0.1);

        assert_ne!(model.inner.weights, before);
    }

    #[test]
    fn test_model_train_epoch_updates_biases() {
        let mut model = ParallelModel::default();
        let before = model.inner.biases.clone();
        let samples = vec![0.1; 32 * NUM_PIXELS];
        let labels = vec![1; 32];

        model.train_epoch(&samples, &labels, 0.1);

        assert_ne!(model.inner.biases, before);
    }
}
