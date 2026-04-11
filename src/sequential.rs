//! A single layer of 10 perceptrons, one per MNIST digit class (0-9).

use crate::data::{NUM_CLASSES, NUM_PIXELS};
use crate::math::{dot, gradient_step, softmax};
use crate::model::Model;

use rand::RngExt;
use serde::{Deserialize, Serialize};

/// Represents an SLP for MNIST classification.
#[derive(Serialize, Deserialize)]
pub struct SequentialModel {
    pub weights: Vec<f32>,
    pub biases: Vec<f32>, // One bias per class, 0-9.
}

impl SequentialModel {
    /// Create a new model with random weights.
    ///
    /// # Returns
    /// - A `SequentialModel` instance with weights randomized from [-0.01, 0.01].
    pub fn new() -> Self {
        let mut rng = rand::rng();

        // Randomize so each perceptron learns differently.
        let weights = (0..NUM_CLASSES * NUM_PIXELS)
            .map(|_| rng.random_range(-0.01..=0.01))
            .collect();

        Self {
            weights,
            biases: vec![0.0; NUM_CLASSES],
        }
    }
}

impl Default for SequentialModel {
    fn default() -> Self {
        Self::new()
    }
}

impl Model for SequentialModel {
    fn forward_pass(&self, pixels: &[f32]) -> Vec<f32> {
        // score_c = dot(weights_c, pixels) + bias_c
        let scores: Vec<f32> = (0..NUM_CLASSES)
            .map(|c| {
                let start = c * NUM_PIXELS;
                let end = start + NUM_PIXELS;
                dot(&self.weights[start..end], pixels) + self.biases[c]
            })
            .collect();

        // Convert raw scores to probabilities using softmax.
        softmax(&scores)
    }

    fn predict(&self, pixels: &[f32]) -> u8 {
        let probabilities = self.forward_pass(pixels);

        // Find the index of the class with the highest probability.
        probabilities
            .iter()
            .enumerate()
            .max_by(|x, y| x.1.total_cmp(y.1))
            .map(|(i, _)| i as u8)
            .unwrap_or(0)
    }

    /// Perform gradient descent on one sample.
    ///
    /// # Formula
    /// ```md
    /// dL/do_i = p_i - y_i
    /// dL/dw   = dL/do_i * x
    /// w       = w - lr * dL/dw
    /// b       = b - lr * dL/do_i
    /// ```
    ///
    /// # Parameters
    /// - `pixels`: A normalized pixel slice of length 784.
    /// - `label`: The correct class label in the range 0-9.
    /// - `learning_rate`: The learning rate value for the gradient descent step.
    ///
    /// # References
    /// - <https://parasdahal.com/softmax-crossentropy/>
    fn train(&mut self, pixels: &[f32], label: u8, learning_rate: f32) {
        let probabilities = self.forward_pass(pixels);
        let mut gradient = vec![0.0f32; NUM_PIXELS];

        for c in 0..NUM_CLASSES {
            // One-hot encode the correct label, 1.0 or 0.0 if false.
            let correct_label = if c == label as usize { 1.0 } else { 0.0 };

            // dL/do_i = p_i - y_i, where p_i is the predicted probability and y_i is the correct label.
            let error = probabilities[c] - correct_label;

            // Reuse the buffer each iteration instead of allocating.
            gradient
                .iter_mut()
                .zip(pixels)
                .for_each(|(g, &pixel)| *g = error * pixel); // dL/dw = dL/do_i * x

            let start = c * NUM_PIXELS;

            // w = w - lr * dL/dw, where lr is the learning rate.
            gradient_step(
                &mut self.weights[start..start + NUM_PIXELS],
                &gradient,
                learning_rate,
            );

            //  b = b - lr * dL/do_i
            self.biases[c] -= learning_rate * error;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_default() {
        let model = SequentialModel::default();

        assert_eq!(model.weights.len(), NUM_CLASSES * NUM_PIXELS);
        assert_eq!(model.biases.len(), NUM_CLASSES);
    }

    #[test]
    fn test_model_forward_length() {
        let model = SequentialModel::default();
        let pixels = vec![0.1; NUM_PIXELS];

        assert_eq!(model.forward_pass(&pixels).len(), NUM_CLASSES);
    }

    #[test]
    fn test_model_predict_range() {
        let model = SequentialModel::default();
        let pixels = vec![0.1; NUM_PIXELS];

        assert!(model.predict(&pixels) <= 9);
    }

    #[test]
    fn test_model_train_updates_weights() {
        let mut model = SequentialModel::default();
        let before = model.weights.clone();
        let pixels = vec![0.1; NUM_PIXELS];

        model.train(&pixels, 1, 0.1);

        assert_ne!(model.weights, before);
    }

    #[test]
    fn test_model_train_updates_biases() {
        let mut model = SequentialModel::default();
        let before = model.biases.clone();
        let pixels = vec![0.1; NUM_PIXELS];

        model.train(&pixels, 1, 0.1);

        assert_ne!(model.biases, before);
    }
}
