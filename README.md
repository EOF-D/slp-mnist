<div align="center">
    <h1>SLP-MNIST</h1>
    Authors:
    <a href="https://github.com/EOF-D">
      EOF-D
    </a>
</div>

## Project Description

SLP-MNIST is a SLP (Single Layer Perceptron) classifier for the MNIST dataset. Implemented in Rust, it trains using gradient descent to minimize the softmax cross-entropy loss with exponential learning rate decay. The model achieves the best accuracy using the sequential version, with a max observed accuracy of ~92.77%.
The project also includes a parallel model which uses mini-batch gradient accumulation (using rayon).

**Evaluation Metrics:**

- Execution Time: total training time
- Classification Accuracy: percentage of correctly classified samples
- Throughput: images classified per second
- Speedup: ratio of sequential to parallel execution time
- Efficiency: speedup divided by number of threads

**Parameters Tested:** number of threads, batch size

## Prerequisites

```text
- Rust version        : rustc 1.85.0+ (edition 2024)
- System requirements : 4GB+ RAM (recommended) + modern multi-core CPU
- Dependencies        : cargo, clap, rand, rayon, serde, serde_json
```

## Setup Instructions

### Dataset

Download the MNIST dataset from [Kaggle](https://www.kaggle.com/datasets/hojjatk/mnist-dataset), then place the related IDX-ubyte files in the `data/` directory at the project's root.

```text
data/
  t10k-images.idx3-ubyte
  t10k-labels.idx1-ubyte
  train-images.idx3-ubyte
  train-labels.idx1-ubyte
```

### Running

Run and train a new model based on default parameters (see `slp_mnist --help`).

```shell
cargo run --release
----------------------------------------------------------------------
model_type=sequential epochs=30 lr=0.01 decay=0.5 step=5 batch_size=32
----------------------------------------------------------------------
epoch   1  9178/10000 (91.78%) lr=0.01000 throughput=127840/s ^
epoch   2  9194/10000 (91.94%) lr=0.01000 throughput=126161/s ^
epoch   3  9206/10000 (92.06%) lr=0.01000 throughput=125814/s ^
epoch   4  9216/10000 (92.16%) lr=0.01000 throughput=127903/s ^
epoch   5  9228/10000 (92.28%) lr=0.00607 throughput=123833/s ^
...
epoch  29  9258/10000 (92.58%) lr=0.00082 throughput=127956/s
epoch  30  9256/10000 (92.56%) lr=0.00050 throughput=125926/s
best epoch: 21 9260/10000 (92.60%) lr=0.00135
total time: 22.58s
----------------------------------------------------------------------
```

Running the binary with parameters matching pre-existing models will run by using saved weights.

```shell
cargo run --release
----------------------------------------------------------------------
model_type=sequential epochs=30 lr=0.01 decay=0.5 step=5 batch_size=32
----------------------------------------------------------------------
loaded from: models/model_sequential_lr0.01_decay0.5_step5_epochs30_batch32.json
test accuracy: 9260/10000 (92.60%)
----------------------------------------------------------------------
```

```shell
cargo run --release -- --help
SLP MNIST classifier

Usage: slp_mnist.exe [OPTIONS]

Options:
      --epochs <EPOCHS>          Number of training epochs [default: 30]
      --lr <LR>                  Initial learning rate [default: 0.01]
      --decay <DECAY>            Exponential decay rate [default: 0.5]
      --step <STEP>              Decay step size (epochs per decay step) [default: 5]
      --model-type <MODEL_TYPE>  Model type to run `sequential`, `parallel`, or `both` [default: sequential]
      --batch-size <BATCH_SIZE>  Batch size (only used by the parallel model) [default: 32]
      --model-path <MODEL_PATH>  Override model path [default: generated from parameters]
  -h, --help                     Print help

cargo run --release -- --epochs 5 --lr 0.1 --decay 0.5 --step 2
--------------------------------------------------------------------
model_type=sequential epochs=5 lr=0.1 decay=0.5 step=2 batch_size=32
--------------------------------------------------------------------
epoch   1  1009/10000 (10.09%) lr=0.10000 throughput=128359/s ^
epoch   2  1009/10000 (10.09%) lr=0.06065 throughput=124478/s
epoch   3  1009/10000 (10.09%) lr=0.06065 throughput=124460/s
epoch   4  1009/10000 (10.09%) lr=0.03679 throughput=126337/s
epoch   5  1009/10000 (10.09%) lr=0.03679 throughput=126354/s
best epoch: 1 1009/10000 (10.09%) lr=0.10000
total time: 3.98s
--------------------------------------------------------------------
```

# References

- [MNIST Dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)
- [Softmax + Cross-Entropy](https://parasdahal.com/softmax-crossentropy/)
- [Softmax Activation](https://www.pinecone.io/learn/softmax-activation/)
- [Learning Rate Decay](https://www.geeksforgeeks.org/machine-learning/learning-rate-decay/)
- [IDX File Format](https://www.fon.hum.uva.nl/praat/manual/IDX_file_format.html)
