# Convolutional Neural Network from Scratch

A 2D CNN for classifying MNIST digits, with every part of the forward and backward pass written by hand using NumPy and SciPy — no PyTorch or TensorFlow. The point of this one was to actually implement convolution, pooling, and their gradients myself instead of calling `nn.Conv2d`.

## How it works

The whole model lives in `src/cnn.py` as a single `CompleteCNN` class:

1. **Convolution** — 4 kernels of size 3x3 slide over the `(1, 28, 28)` input using `scipy.signal.correlate2d`, producing a `(4, 26, 26)` feature map, followed by ReLU.
2. **Max pooling** — a vectorized 2x2 max pool (reshaping into blocks and taking the max, rather than looping over windows) brings it down to `(4, 13, 13)`. The positions of each max are cached as a boolean mask for the backward pass.
3. **Dropout** — 20% of the flattened features are zeroed out during training only.
4. **Dense + softmax** — the flattened `676`-unit vector goes through a single dense layer down to 10 class scores, then softmax.

The backward pass mirrors this in reverse: the softmax + cross-entropy gradient simplifies to `predicted_probabilities - one_hot_label`, which flows back through the dense layer, gets un-dropped-out, gets un-pooled by repeating each gradient value back across its original 2x2 block (using the cached max-position mask), through the ReLU derivative, and finally back through the convolution using `correlate2d` again (`'valid'` mode for the kernel gradient, `'full'` mode for the input gradient).

## Getting started

The MNIST files (`train-images.idx3-ubyte`, `train-labels.idx1-ubyte`, `t10k-images.idx3-ubyte`, `t10k-labels.idx1-ubyte`) already live under `data/MNIST/` in this repo. If you need to re-download them, `src/data_preprocessing.py` points at the [Kaggle MNIST dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset).

```bash
uv sync

# train from scratch — trains one sample at a time, up to 100 epochs,
# stops early once train accuracy hits 100% (or holds above 99% for 10 epochs)
uv run src/train.py

# evaluate on the test set and pop up a window showing misclassified digits
uv run src/test.py

# run inference on your own images — drop grayscale images into src/added_images/
uv run src/predict.py
```

Trained weights are saved to `model/CNN_weights.pkl`.
