Here is a clean, comprehensive, and professional `README.md` for your stock prediction project, styled exactly like your reference example.

---

# Deep Learning LSTM for NVDA Stock Prediction

A custom PyTorch Deep Learning project that predicts the opening price of NVIDIA (`NVDA`) stock using a sequential Long Short-Term Memory (LSTM) network trained on engineered historical market data.

## 📌 Overview

This project implements a sequence-to-scalar architecture designed to process multi-variable historical stock features over a continuous lookback window. Given **50 consecutive trading days** of engineered market indicators, the network predicts the opening price of the next upcoming trading session.

> ⚠️ **Important Data & Pipeline Notice:** Due to an unresolved system multi-threading conflict between the native `yfinance` C-downloader backend and local terminal environments (causing hard `Segmentation fault (core dumped)` errors in raw `.py` scripts), **the data pipeline is managed via Jupyter Notebooks (`.ipynb`)**. Running the downloads within a notebook context bypasses this memory-allocation glitch.

## 🛠️ Model Architecture

Built explicitly using **PyTorch**, the model parses spatial-temporal sequence dynamics from scratch:

* **Input Size:** Takes a 3D feature tensor of size `(Batch, 50, 3)` containing daily engineered features.
* **Recurrent Layer (`StockLSTM`):** A single-layer Long Short-Term Memory network with a hidden layer state of `50` units to retain temporal dependencies over long dependencies without vanishing gradients.
* **Output Layer:** A Linear mapping unit outputting a single scaled scalar value representing the target raw feature (`Next_Open`).

## 📊 Engineered Features

To help the model learn patterns effectively, raw prices are engineered into normalized metrics inside the notebook pipeline:

1. **Volatility:** $\log(\text{High} / \text{Low})$ — tracks daily price range expansion.
2. **Log Returns:** $\log(\text{Close} / \text{Close}_{t-1})$ — standardizes price movement independent of total dollar scale.
3. **Volume:** Min-Max scaled trading volume to standardize liquidity surges.

## 🧰 Tech Stack & Tools

* **Core:** Python, PyTorch
* **Data Retrieval & Analysis:** `yfinance`, Pandas, NumPy
* **Notebook Runner:** Jupyter Notebooks (for data compilation and file extraction)

## 🚀 Key Engineering Highlights

* **Robust Data Alignment:** Fixes traditional lookback bugs by aligning a pre-shifted target parameter (`Next_Open`) precisely with the final chronological timestamp index row of the sequence block.
* **Train/Test Lookback Cushion:** Implemented a boundary overlap mechanism inside the dataset split function, giving the evaluation set a 50-day background context slice so that `eval_loader` never suffers from empty-batch allocations.
* **Isolated Local Inference:** Features an inference mode that bypasses live network scrapers completely, pulling processed telemetry records directly from a local CSV database to predict upcoming opens reliably.

## 📊 Getting Started & Workflow

Because of the network wrapper constraints, you must follow this exact step-by-step pipeline sequence to properly build your databases and run predictions.

### 1. Data Compilation (The Notebook Step)

Before launching the Python CLI, you must populate your tracking files by opening your IDE and executing **both `.ipynb` notebooks**:

* Run the data notebook to pull down NVDA market metrics safely without core-dumping.
* This will successfully export your cleaned, normalized parameters out into `.parent.parent/nvidea_stocks2.csv`.

### 2. Live Prediction Bounds

To run a successful, non-crashing validation check on the model, **you must use target dates between February 20, 2026, and May 1, 2026**.

* The model requires 50 historical trading days *prior* to your target date to construct its timeline array.
* Ensure you attempt predictions **only when the stock market is open** or when a valid trading period is accurately captured within the CSV boundaries.

### 3. Execution

Launch the core command-line application interface to train or predict:

```bash
python main.py

```

* Enter `t` to split your local CSV data, structure training pipelines, and execute backpropagation training over **1,000 epochs**. Weights are auto-saved to `model/model.pt`.
* Enter **any other key** to skip training, load the last saved weight profile, pull the trailing lookback frame from your CSV, and predict the next day's market open in real USD.