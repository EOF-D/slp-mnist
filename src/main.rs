use clap::Parser;
use slp_mnist::data::Dataset;
use slp_mnist::model::Model;

use std::{fs, path::Path, time::Instant};

#[derive(Parser, Debug)]
#[command(name = "slp-mnist", about = "SLP MNIST classifier")]
struct Args {
    /// Number of training epochs.
    #[arg(long, default_value_t = 30)]
    epochs: usize,

    /// Initial learning rate.
    #[arg(long, default_value_t = 0.01)]
    lr: f32,

    /// Exponential decay rate.
    #[arg(long, default_value_t = 0.5)]
    decay: f32,

    /// Decay step size (epochs per decay step).
    #[arg(long, default_value_t = 5)]
    step: i32,

    /// Override model path (default: generated from parameters).
    #[arg(long)]
    model_path: Option<String>,
}

fn main() {
    let args = Args::parse();
    let model_path = args.model_path.unwrap_or_else(|| {
        // Generate a model path based on the training parameters.
        format!(
            "models/model_lr{}_decay{}_step{}_epochs{}.json",
            args.lr, args.decay, args.step, args.epochs
        )
    });

    let total_time = Instant::now();

    // Load the MNIST datasets.
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

    let header = format!(
        "epochs={} lr={} decay={} step={} train={} test={}",
        args.epochs,
        args.lr,
        args.decay,
        args.step,
        train.len(),
        test.len()
    );

    let sep = "-".repeat(header.len());

    println!("{sep}");
    println!("{header}");
    println!("{sep}");

    // Use JSON serialization for model parameters if exists.
    if let Ok(model) = Model::deserialize(&model_path) {
        println!("model={model_path}");
        println!(
            "test accuracy: {:.2}%",
            (0..test.len())
                .filter(|&i| model.predict(test.sample(i)) == test.labels[i])
                .count() as f32
                / test.len() as f32
                * 100.0
        );

        println!("total time: {:>7.2}s", total_time.elapsed().as_secs_f32());
        println!("{sep}");

        return;
    }

    // Create parent directories if they don't exist.
    if let Some(dir) = Path::new(&model_path).parent() {
        fs::create_dir_all(dir).unwrap();
    }

    let mut model = Model::default();
    let mut best_epoch = 0;
    let mut best_correct = 0;
    let mut best_accuracy = 0.0f32;
    let mut best_lr = 0.0f32;

    for epoch in 1..=args.epochs {
        // Exponential decay: https://www.geeksforgeeks.org/machine-learning/learning-rate-decay/
        // Modified with steps.
        let lr = args.lr * (-args.decay * (epoch as i32 / args.step) as f32).exp();

        for i in 0..train.len() {
            model.train(train.sample(i), train.labels[i], lr);
        }

        let correct = (0..test.len())
            .filter(|&i| model.predict(test.sample(i)) == test.labels[i])
            .count();

        let accuracy = correct as f32 / test.len() as f32 * 100.0;
        let marker = if accuracy > best_accuracy { "^" } else { "" };

        println!(
            "epoch {epoch:>3} {correct:>5}/{} ({accuracy:5.2}%) lr={lr:.5} {marker}",
            test.len()
        );

        if accuracy > best_accuracy {
            best_epoch = epoch;
            best_correct = correct;
            best_accuracy = accuracy;
            best_lr = lr;

            // Save the best model immediately.
            model.serialize(&model_path).unwrap();
        }
    }

    println!("{sep}");
    println!(
        "best epoch: {best_epoch} {best_correct}/{} ({best_accuracy:.2}%) lr={best_lr:.5}",
        test.len()
    );

    println!("total time: {:.2}s", total_time.elapsed().as_secs_f32());
    println!("{sep}");
}
