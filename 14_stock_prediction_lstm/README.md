<div align="center">

# 📈 LSTM for NVDA Stock Prediction

**A single-layer LSTM over 50 days of engineered features.**

![Deep Learning](https://img.shields.io/badge/🟣_DEEP_LEARNING-8957e5?style=for-the-badge)

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![yfinance](https://img.shields.io/badge/yfinance-6001D2?style=flat-square)

</div>

---

Predicts NVIDIA's (`NVDA`) next-day move from recent trading history, using a PyTorch LSTM. It reads the last 50 trading days of engineered features and predicts the next day's log return, then converts that back to a dollar price.

## 🧠 How it works

### Data

Data prep lives in two notebooks rather than a plain script:

- `visualise.ipynb` pulls NVDA OHLCV from `yfinance` (2021 → early 2026), engineers three features, and writes `nvidea_stocks.csv` (headerless, for training):
  - **Volatility** — `log(High / Low)`
  - **Log returns** — `log(Close / Close_{t-1})`
  - **Scaled volume** — `Volume / rolling_20day_mean(Volume) - 1`
  - plus the target: next-day log return (`Log_Returns` shifted by -1).
- `get_prediction_sample.ipynb` runs the same feature engineering over a recent window (Jan–June 2026) into `nvidea_stocks_pred.csv`, keeping `Date` and raw `Close` — what `predict.py` reads at inference.

(These were split into notebooks after `yfinance` download issues in plain scripts — worth re-testing if you don't hit the same.) Both CSVs already ship here, so you don't need to rerun them unless you want fresher data.

### Model

`StockLSTM` (`src/stock_lstm.py`) is a single-layer LSTM: input size 3, hidden 50, then `Linear(50 → 25)`, `ReLU`, `Linear(25 → 1)`. It takes a window of daily feature rows and outputs one scalar — the next session's predicted log return.

### Training (`src/train.py`)

`data_preprocessing.make_loader` slices `nvidea_stocks.csv` into 50-day windows (batch 32, no shuffle). Training uses a 70/30 split, `MSELoss`, Adam (lr `0.0001`), 1000 epochs, printing train/validation loss each epoch. Weights save to `model/model.pt` (already included).

### Prediction (`src/predict.py`)

`predict_tomorrow_open()` looks up a target date in `nvidea_stocks_pred.csv`, grabs the 50 days before it plus the target day (51 rows), and feeds that window in. The predicted log return is exponentiated and multiplied by the target day's actual close for a dollar estimate — labeled "tomorrow's opening price," though it's really closer to a next-day close-to-close estimate, since that's what the model was trained on.

The target date must exist in the CSV with ≥50 rows of prior history, or it raises `KeyError`/`ValueError` rather than guessing. It's hardcoded (`2026-05-20`, inside the CSV's `2026-01-30`–`2026-05-29` range) as a default argument; the CLI doesn't prompt for a different one, so call the function directly or edit the default to check another date.

## 🚀 Getting started

```bash
uv sync
uv run main.py
```

You'll be prompted:
- `t` — trains a new model and overwrites `model/model.pt`.
- `p` — runs a prediction for the hardcoded target date and prints the estimate.
- anything else — exits.

## ⚠️ Disclaimer

A portfolio/learning project, not investment advice. A small LSTM over three engineered features isn't a serious market predictor — prices are driven by far more than 50 days of volatility, returns, and volume. Don't trade real money on this.
