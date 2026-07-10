# 🧠 Intel Image Classification with Interactive PyTorch CNN

A highly modular, production-ready implementation of a 2D Convolutional Neural Network (CNN) built using **PyTorch**. This project represents the next step in my Machine Learning Portfolio—transitioning from the first-principles mathematics of a scratch-built NumPy network (Project 11) to engineering scalable deep learning pipelines using industry-standard frameworks, automated hardware acceleration, and dynamic CLI controls.

---

## 🛠️ Key Architectural Elements

* **Dynamic Command-Line Control Loop (`main.py`)**: Features a stateful, interactive user interface that gives real-time runtime choices to initiate hardware-accelerated training blocks, checkpoint models, or load serializations for validation testing.
* **Contracting Funnel Topology (`IntelCNN`)**: Leverages an optimized structural dimension layout, scaling spatial filters sequentially ($3 \to 16 \to 32 \to 64$) while using max-pooling to compact spatial resolution ($150 \times 150 \to 18 \times 18$) for dense linear conversion.
* **Overfitting Mitigation & Regularization**: Integrates dynamic 50% neuron dropout masks into the primary dense layer pathways, enforcing robust feature selection and preventing conceptual memorization during high-capacity scene classification.
* **Universal Hardware Backends**: Automatically identifies and binds to the fastest available local runtime compute environment (`CUDA` for NVIDIA architectures, `MPS` for Apple Silicon metal layers, or optimized fallback threads on native `CPU`).

---

## 📊 Core Pipeline Flow

### 1. The Interactive CLI Architecture

```
[Start Program] ──> Train Model? (y/n) ──> [If 'y': Train, Evaluate Validation & Save?]
                                                                    │
[End Program]   <── Test Model?  (y/n) <── [If 'y': Load Checkpoint & Validate Test Split]

```

### 2. CNN Dimensional Progression

```
Input (3, 150, 150) ──> [ Conv2D + Pool 1 ] ──> (16, 75, 75)
                                                       │
(6, Outputs) <── [ Dense Layer 2 ] <── [ Dense Layer 1 ] <── Flatten Layer <── (64, 18, 18)

```

---

## 📂 Directory Structure

```
12_intel_image_classification/
│
├── src/
│   ├── __init__.py             # Exposes data loader pipelines, models, and execution loops
│   ├── data_loader.py          # ImageFolder configurations and randomized Train/Val splits
│   ├── model.py                # IntelCNN PyTorch nn.Module network topology 
│   ├── engine.py               # Optimized train_model and test_model runtime loops
│   └── utils.py                # State-dict serialization saving, loading, and logging drivers
├── main.py                     # Main interactive orchestration engine script
├── pyproject.toml              # uv-managed project dependencies
└── README.md                   # Project documentation

```

---

### Running the Pipeline

Dependencies are managed with [uv](https://docs.astral.sh/uv/) and are self-contained within this project folder:

```bash
uv sync
```

Orchestrate the modular engine using the main entry-point:

```bash
uv run main.py

```

---

## 🎯 Engineering Insights & Edge Cases Handled

* **Validation-Isolated Partitions (`Val Split`)**: Avoids standard testing dataset leaks by splitting the default dataset dynamically using `torch.utils.data.random_split`. The validation partition monitors out-of-sample generalizability as an analytical checkpoint *during* the training loop.
* **Gradients Accumulation Clear-outs**: Calls `optimizer.zero_grad()` cleanly at the head of every backward sequence step to erase mathematical histories, stopping inter-batch gradient aggregation from destroying optimization directions.
* **State Dict Serialization Safeguards**: Implements isolated model snapshot configurations using PyTorch `.state_dict()` weights-only protocols instead of dumping whole objects, making code maintenance and model version transfers more stable.
* **Stateless Evaluation Transforms**: Explicitly turns off dropout layers and gradient calculation structures via evaluation switches when executing validation tests, maintaining full model integrity when inferring raw test sets.