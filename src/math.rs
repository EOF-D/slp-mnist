//! Math module for machine learning related operations.

use crate::data::{NUM_CLASSES, NUM_PIXELS};

/// Calculate the dot product of two vectors.
///
/// # Parameters
/// - `x`: The first vector.
/// - `y`: The second vector.
///
/// # Returns
/// - The dot product of the two vectors.
///
/// # Panics
/// - Panics if the input vectors have different lengths.
pub fn dot(x: &[f32], y: &[f32]) -> f32 {
    assert_eq!(x.len(), y.len());

    x.iter().zip(y.iter()).map(|(a, b)| a * b).sum()
}

/// Perform a gradient descent step to update weights.
///
/// # Parameters
/// - `weights`: The weights to be updated.
/// - `gradient`: The gradient to apply to the weights.
/// - `learning_rate`: The learning rate to scale the gradient.
///
/// # Panics
/// - Panics if the weights and gradient have different lengths.
pub fn gradient_step(weights: &mut [f32], gradient: &[f32], learning_rate: f32) {
    assert_eq!(weights.len(), gradient.len());

    // w_new = w_old - learning_rate * gradient
    weights
        .iter_mut()
        .zip(gradient)
        .for_each(|(w, g)| *w -= learning_rate * g);
}

/// Apply the softmax function to a vector of raw scores, returning a probability distribution.
///
/// # Formula
///
/// ```md
/// exp_scores_i = exp(z_i)
/// softmax(z_i) = exp_scores_i / sum(exp_scores_j for all j)
/// ```
///
/// # Parameters
/// - `scores`: The raw score vector.
///
/// # Returns
/// - A new vector of probabilities corresponding to each score.
///
/// # References
/// - <https://www.pinecone.io/learn/softmax-activation/>
pub fn softmax(scores: &[f32]) -> Vec<f32> {
    let exp_scores: Vec<f32> = scores.iter().map(|&z| z.exp()).collect();
    let sum: f32 = exp_scores.iter().sum();

    exp_scores.iter().map(|&e| e / sum).collect()
}

/// Compute weight and bias gradients using softmax cross-entropy loss.
///
/// # Formula
/// ```md
/// dL/do_i = p_i - y_i
/// dL/dw   = dL/do_i * x
/// ```
///
/// # Parameters
/// - `pixels`: A normalized pixel slice of length 784.
/// - `probabilities`: The predicted probability distribution over 10 classes.
/// - `label`: The correct class label in the range 0-9.
///
/// # Returns
/// - A tuple `(weight_gradients, bias_gradients)`.
///
/// # References
/// - <https://parasdahal.com/softmax-crossentropy/>
pub fn compute_gradients(pixels: &[f32], probabilities: &[f32], label: u8) -> (Vec<f32>, Vec<f32>) {
    let mut weight_gradients = vec![0.0f32; NUM_CLASSES * NUM_PIXELS];
    let mut bias_gradients = vec![0.0f32; NUM_CLASSES];

    for c in 0..NUM_CLASSES {
        // One-hot encode the correct label, 1.0 or 0.0 if false.
        let correct_label = if c == label as usize { 1.0 } else { 0.0 };

        // dL/do_i = p_i - y_i, where p_i is the predicted probability and y_i is the correct label.
        let error = probabilities[c] - correct_label;

        // dL/dw = dL/do_i * x
        let start = c * NUM_PIXELS;
        weight_gradients[start..start + NUM_PIXELS]
            .iter_mut()
            .zip(pixels)
            .for_each(|(g, &pixel)| *g = error * pixel);

        bias_gradients[c] = error;
    }

    (weight_gradients, bias_gradients)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_normal() {
        assert_eq!(dot(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]), 32.0);
    }

    #[test]
    fn test_dot_zeros() {
        assert_eq!(dot(&[0.0, 0.0, 0.0], &[1.0, 2.0, 3.0]), 0.0);
    }

    #[test]
    #[should_panic]
    fn test_dot_mismatched() {
        dot(&[1.0, 2.0], &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_gradient_step_normal() {
        let mut weights = vec![1.0, 2.0, 3.0];
        let gradient = vec![1.0, 1.0, 1.0];

        gradient_step(&mut weights, &gradient, 0.1);

        assert_eq!(weights, vec![0.9, 1.9, 2.9]);
    }

    #[test]
    fn test_gradient_step_zero_learning_rate() {
        let mut weights = vec![1.0, 2.0, 3.0];
        let gradient = vec![4.0, 5.0, 6.0];

        gradient_step(&mut weights, &gradient, 0.0);

        assert_eq!(weights, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_gradient_step_zero_gradient() {
        let mut weights = vec![1.0, 2.0, 3.0];
        let gradient = vec![0.0, 0.0, 0.0];

        gradient_step(&mut weights, &gradient, 0.1);

        assert_eq!(weights, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    #[should_panic]
    fn test_gradient_step_mismatched() {
        let mut weights = vec![1.0, 2.0];
        let gradient = vec![1.0, 2.0, 3.0];

        gradient_step(&mut weights, &gradient, 0.1);
    }

    #[test]
    fn test_softmax_most_probable() {
        let scores = vec![2.0, 1.0, 3.0, 2.0, 5.0, 9.0, 2.0, 1.0, 3.0, 4.0];
        let probabilities = softmax(&scores);

        let max = probabilities
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);

        // Index 5 (9.0) should be the most probable.
        assert_eq!(probabilities[5], max);
    }

    #[test]
    fn test_compute_gradients_len() {
        let pixels = vec![0.5; NUM_PIXELS];
        let probabilities = vec![0.1; NUM_CLASSES];

        let (weight_gradients, bias_gradients) = compute_gradients(&pixels, &probabilities, 0);

        assert_eq!(weight_gradients.len(), NUM_CLASSES * NUM_PIXELS);
        assert_eq!(bias_gradients.len(), NUM_CLASSES);
    }
}
