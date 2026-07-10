# 🧠 2D Convolutional Neural Network from Scratch

A clean, framework-free implementation of a 2D Convolutional Neural Network (CNN) built entirely using **NumPy** and **SciPy**. This project demonstrates a deep, first-principles understanding of deep learning mathematics, forward/backward propagation, spatial feature extraction, and matrix manipulation—all without relying on high-level frameworks like PyTorch or TensorFlow.

---

## 🛠️ Key Architectural Elements

* **End-to-End Pipeline (`CompleteCNN`)**: Encapsulates the entire model lifecycle within a clean, production-ready class interface containing straightforward `forward()` and `backward()` operations.
* **Vectorized Max Pooling**: Utilizes high-performance, multi-dimensional array reshaping to perform fast max-pooling, cutting down execution time during training.
* **Stable Softmax Activation**: Implements numerical stability trick (maximum-value scaling offsets) to prevent overflow/underflow errors during probability distribution mapping.
* **Dynamic Weight Initialization**: Built to safely handle input dimensions lazily, preventing shape mismatches if input sizes change.

---

## 📊 Core Pipeline Flow

### 1. The Forward Pass

```
Raw Image (1, 28, 28) ──> [ 2D Convolutions ] ──> [ ReLU Activation ]
                                                            │
  Final Probabilities   <── [ Softmax ] <── [ Dense Layer ] <── Flatten Layer

```

### 2. The Backward Pass (Backpropagation)

* **Softmax Gradient**: Computes loss penalties smoothly ($\text{Probability} - 1.0$) for the cross-entropy loss function.
* **Dense Layer Backpropagation**: Computes weight and bias gradients using matrix dot products and applies a dynamic dropout mask.
* **Transposed Convolution Logic**: Routes error gradients back through convolutional layers using SciPy's `'full'` correlation mode, restoring the original $28 \times 28$ spatial dimensions for subsequent iterations.

---

## 📂 Directory Structure

```
11_convolutional_neural_network/
│
├── src/
│   ├── cnn.py                  # Core CNN architecture class
│   ├── data_preprocessing.py   # IDX binary parsing, normalization, & image formatting
│   ├── save_model.py           # Model serialization utilities (save/load weights)
│   ├── train.py                # Model training loop pipeline
│   ├── test.py                 # Evaluation loop & metrics reporting
│   └── predict.py              # CLI inference utility for single custom images
├── added_images/               # Local storage for custom testing images
├── data/MNIST/                 # Local MNIST binary files (see Dataset Installation)
├── pyproject.toml              # uv-managed project dependencies
└── README.md                   # Project documentation

```

---

## ⚙️ Setup & Execution

### 1. Dataset Installation

> 💾 **Data Source:** Download the raw MNIST binary files (`train-images.idx3-ubyte`, `train-labels.idx1-ubyte`, etc.) from the [Official MNIST Database](http://yann.lecun.com/exdb/mnist/) and place them in a local `data/MNIST/` directory.

### 2. Environment Setup

Dependencies are managed with [uv](https://docs.astral.sh/uv/) and are self-contained within this project folder:

```bash
uv sync

```

### 3. Training the Model

Train the network weights from scratch across the training data split:

```bash
uv run src/train.py

```

### 4. Running Evaluations

Evaluate test dataset accuracy, extract performance statistics, and view error readouts:

```bash
uv run src/test.py

```

### 5. Running Inference on Custom Images

Use the newly added prediction pipeline to run single-image inference on your own custom JPEG images:

```bash
uv run src/predict.py

```

---

## 🎯 Engineering Insights & Edge Cases Handled

* **Robust Shape Handling:** Refactored the core matrix operations to safely process odd-shaped images. The pooling layers automatically handle dimensions not perfectly divisible by the pool size without crashing.
* **Input Preprocessing & Normalization:** Standardizes inputs to `(1, 28, 28)` grayscale tensors and normalizes pixel values to `[0.0, 1.0]`. This stabilizes matrix products and prevents gradient explosion (`NaN` values).
* **Lazy Weight Allocation:** The dense layer automatically determines flattening dimensions on the first forward pass, creating a robust, adaptable system design.