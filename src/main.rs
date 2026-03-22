use slp_mnist::data_loader::Dataset;
use std::path::PathBuf;

fn main() {
    let dataset = Dataset::load(PathBuf::from("datasets/mnist_train.csv")).unwrap();

    let first = &dataset.samples[..slp_mnist::data_loader::NUM_PIXELS];
    let min = first.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = first.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    println!("First sample label: {}", dataset.labels[0]);
    println!("First sample pixel range: [{min:.4}, {max:.4}]");
    println!("Expected range: [0.0000, 1.0000]");
}
