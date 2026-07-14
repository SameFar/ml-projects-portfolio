<div align="center">

# 🔢 Convolutional Neural Network from Scratch

**Convolution, pooling, and their gradients — every line by hand in NumPy.**

![Deep Learning](https://img.shields.io/badge/🟣_DEEP_LEARNING-8957e5?style=for-the-badge)

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=flat-square&logo=scipy&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white)

</div>

---

A 2D CNN for MNIST digits with every part of the forward and backward pass written by hand in NumPy and SciPy — no PyTorch or TensorFlow. The point was to implement convolution, pooling, and their gradients myself instead of calling `nn.Conv2d`.

## 🧠 How it works

The whole model is a single `CompleteCNN` class in `src/cnn.py`:

1. **Convolution** — 4 kernels of size 3×3 slide over the `(1, 28, 28)` input via `scipy.signal.correlate2d`, giving a `(4, 26, 26)` feature map, then ReLU.
2. **Max pooling** — a vectorized 2×2 max pool (reshape into blocks, take the max) brings it to `(4, 13, 13)`. Max positions are cached as a boolean mask for backprop.
3. **Dropout** — 20% of the flattened features are zeroed during training only.
4. **Dense + softmax** — the flattened `676`-unit vector goes through one dense layer to 10 class scores, then softmax.

The backward pass mirrors this in reverse: the softmax + cross-entropy gradient simplifies to `predicted_probabilities - one_hot_label`, which flows back through the dense layer, gets un-dropped-out, un-pooled by repeating each gradient across its original 2×2 block (via the cached mask), through the ReLU derivative, and back through the convolution with `correlate2d` (`'valid'` for the kernel gradient, `'full'` for the input gradient).

## 🚀 Getting started

The MNIST files already live under `data/MNIST/`. To re-download them, `src/data_preprocessing.py` points at the [Kaggle MNIST dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset).

```bash
uv sync

# train from scratch — one sample at a time, up to 100 epochs,
# stops early once train accuracy hits 100% (or holds above 99% for 10 epochs)
uv run -m src.train

# evaluate on the test set and pop up a window showing misclassified digits
uv run -m src.test

# run inference on your own images — drop grayscale images into src/added_images/
uv run -m src.predict
```

Trained weights are saved to `model/CNN_weights.pkl`.
