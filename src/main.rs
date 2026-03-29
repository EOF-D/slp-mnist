use slp_mnist::data_loader::Dataset;
use slp_mnist::model::Model;

use std::{fs, path::Path, time::Instant};

const LEARNING_RATE: f32 = 0.01;
const NUM_EPOCHS: usize = 10;
const MODEL_PATH: &str = "models/model.json";

fn main() {
    let total_time = Instant::now();

    let train = Dataset::load(
        "datasets/train-images.idx3-ubyte",
        "datasets/train-labels.idx1-ubyte",
    )
    .unwrap();

    let test = Dataset::load(
        "datasets/t10k-images.idx3-ubyte",
        "datasets/t10k-labels.idx1-ubyte",
    )
    .unwrap();

    println!("Training sample count : {}", train.len());
    println!("Test sample count     : {}", test.len());
    println!();

    // Use JSON serialization for model parameters if exists.
    if let Ok(model) = Model::deserialize(MODEL_PATH) {
        println!("Loaded model parameters from {MODEL_PATH}");
        println!(
            "Test accuracy: {:.2}%",
            (0..test.len())
                .filter(|&i| model.predict(test.sample(i)) == test.labels[i])
                .count() as f32
                / test.len() as f32
                * 100.0
        );

        return;
    }

    let mut model = Model::default();
    for epoch in 1..=NUM_EPOCHS {
        for i in 0..train.len() {
            model.train(train.sample(i), train.labels[i], LEARNING_RATE);
        }

        let correct = (0..test.len())
            .filter(|&i| model.predict(test.sample(i)) == test.labels[i])
            .count();

        let accuracy = correct as f32 / test.len() as f32 * 100.0;
        println!("Epoch {epoch}: {correct} / {} ({accuracy:.2}%)", test.len());
    }

    // Create parent directory if it doesn't exist before saving the model.
    if let Some(dir) = Path::new(MODEL_PATH).parent() {
        fs::create_dir_all(dir).unwrap();
    }

    model.serialize(MODEL_PATH).unwrap();

    println!();
    println!("Total time: {:.2}s", total_time.elapsed().as_secs_f32());
}
