# Multi-Head Fashion Recommendation Neural Network

This folder contains a neural network designed to predict multiple fashion recommendations simultaneously (such as recommended colors, materials, patterns, and metals).

It features two separate implementations: a custom version built entirely from scratch using raw **NumPy**, and a modernized version built using **PyTorch**.

---

## 🧠 Why Both? (Portfolio Reasoning)

This project contains two separate code tracks to demonstrate a deep understanding of machine learning from first principles up to production deployment:

* **The NumPy Implementation:** Demonstrates a fundamental understanding of underlying ML mathematics. It showcases the ability to manually write forward propagation, execute the chain rule via backward propagation across a multi-head network, track individual losses, and implement manual gradient descent without framework abstractions.
* **The PyTorch Implementation:** Demonstrates modern, industry-standard engineering. It transitions the manual math into optimized framework code using dynamic computational graphs (`autograd`), native optimizers, and automated weight initializations.

---

## 📂 Project Structure

### 🌐 NumPy Production API (`/api`)

* `neural_network.py` — From-scratch NumPy model architecture.
* `app.py` — Vercel-ready FastAPI application for production inference.
* `model/fashion_nn_weights.pkl` — Saved weights loaded by the production API.
* `conversion.py` / `data_preprocessing.py` / `pydantic_model.py` — Production helper scripts for processing incoming features and mapping outputs.

### 🧪 PyTorch Development (`/src`)

* `neural_network.py` — Modernized PyTorch implementation.
* `torch_train.py` — Central training script that optimizes the PyTorch model and saves weight configurations for both tracks.
* `torch_app.py` — Local FastAPI application for testing the PyTorch backend.
* `model/torch_model.pth` — Saved PyTorch weights.
* `data_visualization.ipynb` — Jupyter Notebook for EDA and data processing.
* `conversion.py` / `data_preprocessing.py` / `pydantic_model.py` — Duplicate helper scripts for local development stability.

---

## 🚀 How It Works

1. **Shared Hidden Layer:** The network uses a shared hidden layer with a ReLU activation function to learn base features from the inputs.
2. **Multi-Head Outputs:** The hidden layer branches out into 6 separate output "heads" targeting specific categorical recommendations.
3. **Weight Initialization:** Uses He/Kaiming initialization for the ReLU hidden layer and Xavier/Glorot initialization for the output heads to keep variance stable.
4. **Loss Functions:** Uses Categorical Cross-Entropy for multi-class heads and Binary Cross-Entropy for independent classification heads.

---

## 🛠️ How to Run Locally

### 1. Installation

Dependencies are managed with [uv](https://docs.astral.sh/uv/) and are self-contained within this project folder:

```bash
uv sync

```

> Note: `api/requirements.txt` is kept separately since Vercel's deployment builder reads it directly and does not use uv.

### 2. Training the Network

The central PyTorch training script handles the optimization loop for 2000 epochs, tracks individual head losses, handles learning rate decay, and saves weights to both the development path (`src/model/`) and the production path (`api/model/`):

```bash
uv run src/torch_train.py

```

### 3. Running the Local API Servers

You can spin up either backend to test predictions:

```bash
# Run the NumPy Production Server (Port 8000)
uv run api/app.py
uv run uvicorn api.app:app --host 127.0.0.1 --port 8000 --reload

# Run the PyTorch Development Server (Port 8001)
uv run src/torch_app.py

```

---

## 🌐 Production Deployment (Vercel Serverless)

To keep cloud deployments hyper-fast, lightweight, and completely free of cold-start performance penalties, **the NumPy track is deployed to Vercel Serverless**. This completely bypasses the massive file-size restrictions imposed by framework heavyweights like PyTorch.

By setting the Vercel **Root Directory** setting directly to this `nn/` folder, the configuration routes all incoming cloud traffic straight to the optimized NumPy inference engine inside `api/app.py`.

### Testing Live Predictions

Send a POST request to your live Vercel endpoint suffix: `/predict` using the structured JSON data defined in your `FashionInput` schema to receive mapped text recommendations instantly.

---

## 💡 Engineering Reflection & Takeaways

In hindsight, given the structured and deterministic nature of the current dataset, a simple rule-based search function or a database lookup table could achieve the same results with zero training time and 100% accuracy.

However, this architecture was intentionally engineered as a neural network to achieve two primary goals:

1. **Architectural Complexity:** To demonstrate mastery over multi-task learning networks, custom backpropagation mechanics across separate output heads, and production-grade framework migrations.
2. **Future Generalization:** A database search function breaks the moment user input is noisy, incomplete, or fuzzy. This neural network provides a flexible backbone capable of scaling to non-linear real-world data (such as raw RGB values, missing profile fields, or continuous user metrics) where hardcoded `if/else` logic falls apart.