<div align="center">
    <h1>SLP-MNIST</h1>
</div>

## Running

Run and train a new model based on default parameters (see `slp_mnist --help`).

```shell
cargo run --release
epochs=30 lr=0.01 decay=0.5 step=5 train=60000 test=10000
epoch 1: 9160 / 10000 (91.60%) learning rate: 0.01000 (decay: 0.5)
epoch 2: 9198 / 10000 (91.98%) learning rate: 0.01000 (decay: 0.5)
epoch 3: 9214 / 10000 (92.14%) learning rate: 0.01000 (decay: 0.5)
...
epoch 28: 9259 / 10000 (92.59%) learning rate: 0.00082 (decay: 0.5)
epoch 29: 9260 / 10000 (92.60%) learning rate: 0.00082 (decay: 0.5)
epoch 30: 9260 / 10000 (92.60%) learning rate: 0.00050 (decay: 0.5)
total time: 20.39s
best epoch 17: 9274 / 10000 (92.74%) learning rate: 0.00223
```

Running the binary with parameters matching pre-existing models will run by using saved weights.

```shell
cargo run --release
epochs=30 lr=0.01 decay=0.5 step=5 train=60000 test=10000
loaded from: models/model_lr0.01_decay0.5_step5_epochs30.json
test accuracy: 92.74%
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
      --model-path <MODEL_PATH>  Override model path (default: generated from parameters)
  -h, --help                     Print help

cargo run --release -- --epochs 5 --lr 0.1 --decay 0.5 --step 2
epochs=5 lr=0.1 decay=0.5 step=2 train=60000 test=10000
epoch 1: 1009 / 10000 (10.09%) learning rate: 0.10000 (decay: 0.5)
epoch 2: 1009 / 10000 (10.09%) learning rate: 0.06065 (decay: 0.5)
epoch 3: 1009 / 10000 (10.09%) learning rate: 0.06065 (decay: 0.5)
epoch 4: 1009 / 10000 (10.09%) learning rate: 0.03679 (decay: 0.5)
epoch 5: 1009 / 10000 (10.09%) learning rate: 0.03679 (decay: 0.5)
total time: 3.58s
best epoch 1: 1009 / 10000 (10.09%) learning rate: 0.10000
```

## Dataset

- [MNIST Dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)

## Team Members

- [Andy Zheng](https://github.com/EOF-D) - Maintainer
