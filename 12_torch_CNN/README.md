# Torch CNN — Intel Image Classification

A PyTorch CNN that sorts natural scene photos into six categories (buildings, forest, glacier, mountain, sea, street), trained on the [Intel Image Classification dataset](https://www.kaggle.com/datasets/puneet6060/intel-image-classification). Where [project 11](../11_convolutional_neural_network) is a hand-written CNN, this one is the same idea built with an actual framework — `nn.Module`, `autograd`, real optimizers.

## How it works

`src/model.py` defines `IntelCNN`: three conv + max-pool blocks (3 -> 16 -> 32 -> 64 channels, all 3x3 kernels with padding, ReLU after each), which shrink a `150x150` RGB image down to `64x18x18`. That gets flattened into a dense layer of 512 units with 50% dropout, then a final dense layer down to 6 class scores.

`src/data_load.py` downloads the dataset via `kagglehub` and builds three loaders: an 80/20 train/validation split (`get_train_val_loaders`) and a separate test loader (`get_test_loader`). `src/train.py` trains with Adam (`lr=0.001`) and cross-entropy loss for 10 epochs by default, logging train loss, validation loss, and validation accuracy after each epoch. `src/test.py` runs the same evaluation logic against the held-out test set.

`main.py` at the project root ties it together into a small interactive loop: it asks whether to train (and if so, whether to save the result to `model/model.pth` via `state_dict()`), then asks whether to test (loading `model/model.pth` and reporting accuracy on the test split). It picks the fastest available device automatically — CUDA, then Apple's MPS, then CPU.

## Getting started

The first run downloads the dataset through `kagglehub`, which may prompt you to authenticate with a Kaggle account.

```bash
uv sync
uv run main.py
```

You'll be prompted interactively:

```
Train model? y/n:
Save trained model? y/n:
Test model? y/n:
```
