# Character-Level RNN for Name Origin Classification

A custom PyTorch Deep Learning project that predicts the country of origin (1 of 18 languages) for a given surname using a character-level Recurrent Neural Network (RNN).

## 📌 Overview

Instead of using high-level embeddings or heavy pre-trained models, this project processes text at the **character level from scratch**. It normalizes raw, accented text into standard ASCII, maps characters to one-hot encoded tensors, and feeds them sequentially through a custom RNN to classify the name's language of origin (e.g., `Nakamura` $\rightarrow$ `Japanese`).

## 🛠️ Model Architecture

Built explicitly using **PyTorch**, the model handles sequence data without third-party abstraction:

* **Input Layer:** Takes a one-hot encoded tensor of size `58` (representing unique allowed characters).
* **Recurrent Layer (`nn.RNN`):** A single-layer standard RNN with a hidden size of `128` to maintain sequential context.
* **Output Layer:** A Linear layer mapped to `18` units (representing the 18 target language classes) followed by a `LogSoftmax` activation function.

## 🧰 Tech Stack & Tools

* **Core:** Python, PyTorch
* **Data Processing:** Standard `unicodedata` for text normalization
* **Evaluation:** Matplotlib (loss curve tracking), Standard Python `logging` library

## 🚀 Key Engineering Highlights

* **Robust Custom Dataset:** Implemented a custom PyTorch `Dataset` wrapper (`NamesDataset`) that parses `.txt` files on the fly, dynamically tracks the unique classes, and caches text and labels into tensors.
* **Custom Batching & Collate:** Utilized a tailored `DataLoader` configuration with custom batching logic to safely step through individual sequences without uniform padding overcomplications.
* **Text Sanitization Pipeline:** Built a custom pipeline to normalize Unicode strings down to standard ASCII, handling out-of-vocabulary characters gracefully.
* **Solid Training Hygiene:** Features a proper validation split (`85/15`), gradient clipping (`clip_grad_norm_`) to prevent exploding gradients, and automated model checkpointing (`.pt` exports).

## 📊 Getting Started

### 1. Installation & Data

Dependencies are managed with [uv](https://docs.astral.sh/uv/) and are self-contained within this project folder:
```bash
uv sync
```
Target `.txt` files already live inside `data/names/`, one file per label (e.g., `Arabic.txt`, `Italian.txt`).

### 2. Execution

Run the main script to interface with the CLI wrapper:

```bash
uv run main.py

```

* Press `y` to execute training over 45 epochs. It will automatically log average batch loss, compute validation accuracy, and plot your training curve.
* Press `n` to enter inference mode and guess the origin of any input string using your saved weights.