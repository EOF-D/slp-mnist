use clap::Parser;
use serde::{Serialize, de::DeserializeOwned};

use slp_mnist::args::Args;
use slp_mnist::data::Dataset;
use slp_mnist::parallel::ParallelModel;
use slp_mnist::sequential::SequentialModel;

use std::{fs, fs::File, time::Instant};

#[rustfmt::skip]
fn main() {
    let args = Args::parse();

    let header = args.to_string();
    let sep = "-".repeat(header.len());
    println!("{sep}\n{header}\n{sep}");

    // Create the models directory if it doesn't exist.
    fs::create_dir_all("models").expect("could not create models directory");

    match args.model_type.as_str() {
        "sequential" => { run_sequential(&args); },
        "parallel" => { run_parallel(&args); },
        "both" => {
            let sequential_time = run_sequential(&args);
            println!("{sep}");

            let parallel_time = run_parallel(&args);
            let speedup = sequential_time / parallel_time;
            let threads = rayon::current_num_threads();

            println!("{sep}");
            println!("speedup: {speedup:.2}x");
            println!(
                "efficiency: {:.1}% ({threads} threads)",
                speedup / threads as f32 * 100.0
            );
        }

        _ => panic!("invalid model type: {}", args.model_type),
    }

    println!("{sep}");
}

fn load_model<M: DeserializeOwned>(path: &str) -> Option<M> {
    File::open(path)
        .ok()
        .and_then(|f| serde_json::from_reader(f).ok())
}

fn save_model<M: Serialize>(model: &M, path: &str) {
    serde_json::to_writer(File::create(path).unwrap(), model).unwrap();
}

fn print_epoch(
    epoch: usize,
    correct: usize,
    total: usize,
    accuracy: f32,
    lr: f32,
    throughput: f32,
    is_best: bool,
) {
    let marker = if is_best { "^" } else { "" };
    println!(
        "epoch {epoch:>3} {correct:>5}/{total} ({accuracy:5.2}%) lr={lr:.5} throughput={throughput:.0}/s {marker}",
    );
}

fn print_best(epoch: usize, correct: usize, total: usize, accuracy: f32, lr: f32) {
    println!("best epoch: {epoch} {correct}/{total} ({accuracy:.2}%) lr={lr:.5}");
}

fn run_sequential(args: &Args) -> f32 {
    let total = Instant::now();

    // Load the MNIST datasets.
    let train = Dataset::load(
        "data/train-images.idx3-ubyte",
        "data/train-labels.idx1-ubyte",
    )
    .unwrap();

    let test = Dataset::load("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte").unwrap();

    let model_path = args.model_path("sequential");

    // Use JSON serialization for model parameters if exists.
    if let Some(model) = load_model::<SequentialModel>(&model_path) {
        let correct = (0..test.len())
            .filter(|&i| model.predict(test.sample(i)) == test.labels[i])
            .count();

        let accuracy = correct as f32 / test.len() as f32 * 100.0;

        println!("loaded from: {model_path}");
        println!("test accuracy: {correct}/{} ({accuracy:.2}%)", test.len());

        return total.elapsed().as_secs_f32();
    }

    let mut model = SequentialModel::default();
    let mut best = (0, 0, 0.0f32, 0.0f32); // (epoch, correct, accuracy, lr)

    for epoch in 1..=args.epochs {
        let lr = args.lr(epoch);
        for i in 0..train.len() {
            model.train(train.sample(i), train.labels[i], lr);
        }

        let timer = Instant::now();

        let correct = (0..test.len())
            .filter(|&i| model.predict(test.sample(i)) == test.labels[i])
            .count();

        let accuracy = correct as f32 / test.len() as f32 * 100.0;
        let throughput = test.len() as f32 / timer.elapsed().as_secs_f32();
        let is_best = accuracy > best.2;

        print_epoch(
            epoch,
            correct,
            test.len(),
            accuracy,
            lr,
            throughput,
            is_best,
        );

        if is_best {
            best = (epoch, correct, accuracy, lr);
            save_model(&model, &model_path);
        }
    }

    let total_time = total.elapsed().as_secs_f32();

    print_best(best.0, best.1, test.len(), best.2, best.3);
    println!("total time: {total_time:.2}s");

    total_time
}

fn run_parallel(args: &Args) -> f32 {
    let total = Instant::now();

    // Load the MNIST datasets.
    let train = Dataset::load(
        "data/train-images.idx3-ubyte",
        "data/train-labels.idx1-ubyte",
    )
    .unwrap();

    let test = Dataset::load("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte").unwrap();

    let model_path = args.model_path("parallel");

    // Use JSON serialization for model parameters if exists.
    if let Some(model) = load_model::<ParallelModel>(&model_path) {
        let accuracy = model.batch_test(&test.samples, &test.labels);
        let correct = (accuracy / 100.0 * test.len() as f32).round() as usize;

        println!("loaded from: {model_path}");
        println!("test accuracy: {correct}/{} ({accuracy:.2}%)", test.len());

        return total.elapsed().as_secs_f32();
    }

    let mut model = ParallelModel::new(args.batch_size);
    let mut best = (0, 0, 0.0f32, 0.0f32); // (epoch, correct, accuracy, lr)

    for epoch in 1..=args.epochs {
        let lr = args.lr(epoch);
        model.train_epoch(&train.samples, &train.labels, lr);

        let timer = Instant::now();

        let accuracy = model.batch_test(&test.samples, &test.labels);
        let correct = (accuracy / 100.0 * test.len() as f32).round() as usize;
        let throughput = test.len() as f32 / timer.elapsed().as_secs_f32();
        let is_best = accuracy > best.2;

        print_epoch(
            epoch,
            correct,
            test.len(),
            accuracy,
            lr,
            throughput,
            is_best,
        );

        if is_best {
            best = (epoch, correct, accuracy, lr);
            save_model(&model, &model_path);
        }
    }

    let total_time = total.elapsed().as_secs_f32();
    print_best(best.0, best.1, test.len(), best.2, best.3);
    println!("total time: {total_time:.2}s");

    total_time
}
