<div align="center">

# 🏞️ Torch CNN — Intel Image Classification

**The same idea as project 11, this time with a real framework.**

![Deep Learning](https://img.shields.io/badge/🟣_DEEP_LEARNING-8957e5?style=for-the-badge)

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Kaggle](https://img.shields.io/badge/kagglehub-20BEFF?style=flat-square&logo=kaggle&logoColor=white)

</div>

---

A PyTorch CNN that sorts natural-scene photos into six categories (buildings, forest, glacier, mountain, sea, street) on the [Intel Image Classification dataset](https://www.kaggle.com/datasets/puneet6060/intel-image-classification). Where [project 11](../11_convolutional_neural_network) is a hand-written CNN, this is the same idea with `nn.Module`, `autograd`, and real optimizers.

## 🧠 How it works

`src/model.py` defines `IntelCNN`: three conv + max-pool blocks (3 → 16 → 32 → 64 channels, 3×3 kernels with padding, ReLU each), shrinking a `150×150` RGB image to `64×18×18`. That flattens into a 512-unit dense layer with 50% dropout, then a final dense layer to 6 class scores.

`src/data_load.py` downloads the dataset via `kagglehub` and builds an 80/20 train/validation split (`get_train_val_loaders`) plus a separate test loader. `src/train.py` trains with Adam (`lr=0.001`) and cross-entropy for 10 epochs by default, logging train/validation loss and validation accuracy each epoch; `src/test.py` runs the same eval against the held-out test set.

`main.py` ties it into a small interactive loop: train (optionally saving to `model/model.pth`), then test (loading it back and reporting accuracy). It auto-picks the fastest device — CUDA, then Apple MPS, then CPU.

## 🚀 Getting started

The first run downloads the dataset through `kagglehub`, which may prompt for Kaggle authentication.

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
