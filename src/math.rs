//! Math module for machine learning related operations.

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
/// - https://www.pinecone.io/learn/softmax-activation/
pub fn softmax(scores: &[f32]) -> Vec<f32> {
    let exp_scores: Vec<f32> = scores.iter().map(|&z| z.exp()).collect();
    let sum: f32 = exp_scores.iter().sum();

    exp_scores.iter().map(|&e| e / sum).collect()
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
}
