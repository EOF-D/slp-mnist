//! Model trait for concrete MNIST classifiers.

pub const NUM_CLASSES: usize = 10;

/// Shared interface for concrete MNIST classifier models.
pub trait Model {
    /// Compute the probability distribution for a single sample.
    ///
    /// # Parameters
    /// - `pixels`: A normalized pixel slice of length 784.
    ///
    /// # Returns
    /// - A vector of 10 probabilities, one per digit class.
    fn forward_pass(&self, pixels: &[f32]) -> Vec<f32>;

    /// Predict the digit class for a single sample.
    ///
    /// # Parameters
    /// - `pixels`: A normalized pixel slice of length 784.
    ///
    /// # Returns
    /// - The predicted digit class in the range 0-9.
    fn predict(&self, pixels: &[f32]) -> u8;

    /// Perform a single training step on one sample.
    ///
    /// # Parameters
    /// - `pixels`: A normalized pixel slice of length 784.
    /// - `label`: The correct class label in the range 0-9.
    /// - `learning_rate`: The learning rate value for the gradient descent step.
    fn train(&mut self, pixels: &[f32], label: u8, learning_rate: f32);
}
