use slp_mnist::data_loader::Dataset;

fn main() {
    let dataset = Dataset::load(
        "datasets/train-images.idx3-ubyte",
        "datasets/train-labels.idx1-ubyte",
    )
    .unwrap();

    let first = &dataset.samples[..slp_mnist::data_loader::NUM_PIXELS];
    let min = first.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = first.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    println!("First sample label: {}", dataset.labels[0]);
    println!("First sample pixel range: [{min:.4}, {max:.4}]");
    println!("Expected range: [0.0000, 1.0000]");
}
