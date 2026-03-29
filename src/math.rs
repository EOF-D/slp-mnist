//! Math module for machine learning related operations.

/// Calculate the dot product of two vectors.
///
/// # Parameters
/// - `x`: The first vector.
/// - `y`: The second vector.
///
/// # Returns
/// - The dot product of the two vectors.
pub fn dot(x: &[f32], y: &[f32]) -> f32 {
    assert_eq!(x.len(), y.len());

    x.iter().zip(y.iter()).map(|(a, b)| a * b).sum()
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
}
