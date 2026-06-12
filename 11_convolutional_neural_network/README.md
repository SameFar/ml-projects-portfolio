# 🧠 2D Convolutional Neural Network from Scratch

A first-principles, framework-less implementation of a 2D Convolutional Neural Network (CNN) built entirely in **NumPy** and **SciPy**. This project demonstrates the explicit matrix calculus, spatial manipulation, and tensor adjustments required to build, train, and test an image classification pipeline on the MNIST dataset without relying on PyTorch or TensorFlow.

---

## 🛠️ Key Architectural Elements

* **Unified Network Pipeline (`CompleteCNN`)**: Consolidates the complete deep learning lifecycle into a clean, single-class interface (`forward(x_sample)` and `backward(y_label)`).
* **Valid 2D Cross-Correlation**: Leverages SciPy's spatial tools to scan multi-channel inputs and collapse spatial coordinates down into localized high-dimensional feature maps.
* **Integrated ReLU Activation**: Embedded directly within the feature extractor to introduce non-linear mapping boundaries while preserving pre-activation values for exact gradient matching.
* **Flattening / Spatial Reconstruction Bridge**: Dynamically translates 3D feature arrays into 1D vectors for classification, and seamlessly reshapes downstream error gradients back into 3D tensors during backpropagation.
* **Stable Softmax Classifier**: Maps raw dense output logs into scaled percentages between $0\%$ and $100\%$, utilizing maximum-value scaling offsets to mitigate numerical overflow errors.

---

## 📊 Core Pipeline Flow

### 1. The Forward Pass

```
Raw Image (1, 28, 28) ──> [ 2D Convolutions ] ──> Raw Feature Maps
                                                        │
  1D Logits Vector    <── [ Dense Layer ] <── Flatten 3D <── [ ReLU Activation ]
         │
         └──> [ Softmax Function ] ──> Class Probabilities (0-9)

```

### 2. The Backward Pass (Weight Updates)

* **Softmax Error Calculation**: Computes the baseline penalty vectors by adjusting the active target distribution index ($\text{Probability} - 1.0$).
* **Dense Matrix Chain Rule**: Calculates weight adjustments through outer dot products while pushing incoming blame vectors back toward the flattening boundary.
* **The `'full'` Correlation Reversal**: To route errors back through the convolutional layers, the script implements a specialized `'full'` padding mode. This mathematically inflates the shrunk gradient grids back into the exact $28 \times 28$ spatial size of the original input.

---

## 📂 Directory Structure

```
11_convolutional_neural_network/
│
├── src/
│   ├── cnn.py                  # Single-class CompleteCNN architecture
│   ├── data_preprocessing.py   # IDX binary parsing, normalization, & OpenCV rendering
│   ├── save_model.py           # Serialized weight freezing (save_model/load_model)
│   ├── train.py                # Model training execution pipeline
│   └── test.py                 # Weight unfreezing, evaluation loop, & failure display
├── requirements.txt            # Project environment constraints
└── README.md                   # Project documentation

```

---

## ⚙️ Setup & Execution

### 1. Dataset Dependency Note

> 💾 **Data Source:** Raw MNIST data files are not hosted in this repository due to footprint limitations. Before executing, download the binary files (`train-images.idx3-ubyte`, `train-labels.idx1-ubyte`, `t10k-images.idx3-ubyte`, `t10k-labels.idx1-ubyte`) from the [Official MNIST Database](http://yann.lecun.com/exdb/mnist/) and extract them into your local `data/MNIST/` repository directory path.

### 2. Installation

Initialize dependencies within an isolated project virtual environment:

```bash
pip install -r requirements.txt

```

### 3. Training the Model

Run the core pipeline to train the weights across your localized training splits:

```bash
python src/train.py

```

### 4. Running Evaluations & Visualizing Failures

Execute the test loader script to pull down your saved `CNN_weights.pkl` matrix values, evaluate test dataset accuracy, and isolate misclassified samples.

Press `q` within the OpenCV window frame to cycle through error readouts:

```bash
python src/test_loader.py

```

---

## 🎯 Validation Metrics & Insights

* **The Squeeze Transformation**: Input arrays must be explicitly structured as 4D tensors `(Samples, Channels, Height, Width)` at initialization. Passing flat or raw 2D pixel strips collapses the spatial alignment indices, breaking the internal multi-channel loop routines.
* **Value Range Controls**: Grayscale inputs are normalized down to float vectors bounded between `0.0` and `1.0`. Removing this division step causes the dot products inside the dense layers to instantly blow up, resulting in catastrophic gradient explosion (`NaN` outputs).
* **Failure Windowing**: The tracking script appends true versus predicted mismatch indicators directly into the rendering title bar (`"True: 7 | Pred: 2"`), enabling granular tracking of handwriting ambiguities that trip up the network.
