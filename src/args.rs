//! CLI argument parser for the SLP-MNIST pipeline.

use clap::Parser;
use std::fmt;

#[derive(Parser, Debug)]
#[command(name = "slp-mnist", about = "SLP MNIST classifier")]
pub struct Args {
    /// Number of training epochs.
    #[arg(long, default_value_t = 30)]
    pub epochs: usize,

    /// Initial learning rate.
    #[arg(long, default_value_t = 0.01)]
    pub lr: f32,

    /// Exponential decay rate.
    #[arg(long, default_value_t = 0.5)]
    pub decay: f32,

    /// Decay step size (epochs per decay step).
    #[arg(long, default_value_t = 5)]
    pub step: i32,

    /// Model type to run `sequential`, `parallel`, or `both`.
    #[arg(long, default_value = "sequential")]
    pub model_type: String,

    /// Batch size (only used by the parallel model).
    #[arg(long, default_value_t = 32)]
    pub batch_size: usize,

    /// Override model path (default: generated from parameters).
    #[arg(long)]
    pub model_path: Option<String>,
}

impl Args {
    /// Returns the model save path based on model type
    ///
    /// # Parameters
    /// - `model_type`: The type of model.
    ///
    /// # Returns
    /// - The model save path.
    /// .
    pub fn model_path(&self, model_type: &str) -> String {
        self.model_path.clone().unwrap_or_else(|| {
            format!(
                "models/model_{}_lr{}_decay{}_step{}_epochs{}_batch{}.json",
                model_type, self.lr, self.decay, self.step, self.epochs, self.batch_size
            )
        })
    }

    /// Learning rate using stepped exponential decay.
    ///
    /// # Parameters
    /// - `epoch`: The epoch number to use.
    ///
    /// # Returns
    /// - Learning rate for the given epoch.
    ///
    /// # References
    /// - <https://www.geeksforgeeks.org/machine-learning/learning-rate-decay/>
    pub fn lr(&self, epoch: usize) -> f32 {
        self.lr * (-self.decay * (epoch as i32 / self.step) as f32).exp()
    }
}

impl fmt::Display for Args {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "model_type={} epochs={} lr={} decay={} step={} batch_size={}",
            self.model_type, self.epochs, self.lr, self.decay, self.step, self.batch_size,
        )
    }
}
