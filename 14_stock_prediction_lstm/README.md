# LSTM for NVDA Stock Prediction

This project tries to predict NVIDIA's (`NVDA`) next-day stock movement from its recent trading history, using an LSTM in PyTorch. It looks at the last 50 trading days of engineered features and predicts the next day's log return, which then gets converted back into a dollar price.

## How it works

### Data

Data prep happens in two Jupyter notebooks rather than a plain script:

- `visualise.ipynb` pulls NVDA's OHLCV history from `yfinance` (2021 through early 2026), engineers three features, and writes them out to `nvidea_stocks.csv` (headerless, used for training):
  - **Volatility**: `log(High / Low)`
  - **Log returns**: `log(Close / Close_{t-1})`
  - **Scaled volume**: `Volume / rolling_20day_mean(Volume) - 1`
  - plus a training target column, next-day log return (`Log_Returns` shifted by -1).
- `get_prediction_sample.ipynb` does the same feature engineering over a more recent window (Jan-June 2026) and writes `nvidea_stocks_pred.csv`, keeping the `Date` and raw `Close` columns too — this is what `predict.py` reads from at inference time.

(I originally split this out into notebooks after running into `yfinance` download problems in plain scripts on my machine — worth re-testing if you don't hit the same issue.) Both CSVs already ship in this folder, so you don't need to rerun the notebooks unless you want fresher data.

### Model

`StockLSTM` (`src/stock_lstm.py`) is a single-layer LSTM: input size 3 (volatility, log return, scaled volume), hidden size 50, followed by `Linear(50 → 25)`, `ReLU`, `Linear(25 → 1)`. It takes a window of daily feature rows and outputs one scalar: the predicted log return for the next session.

### Training (`src/train.py`)

`data_preprocessing.make_loader` slices `nvidea_stocks.csv` into 50-day windows to predict the log-return target at the end of each window (batch size 32, no shuffling). Training uses a 70/30 train/validation split, `MSELoss`, Adam (lr `0.0001`), for 1000 epochs, printing train/validation loss every epoch. Weights are saved to `model/model.pt` (already included, from a prior run).

### Prediction (`src/predict.py`)

`predict_tomorrow_open()` looks up a target date in `nvidea_stocks_pred.csv`, grabs the 50 trading days before it plus the target day itself (51 rows), and feeds that window into the model. The predicted log return is exponentiated and multiplied by the target day's actual close price to get a dollar estimate — the code labels this "tomorrow's opening price," though the value it's actually reconstructing is closer to a next-day close-to-close estimate, since that's what the model was trained to predict.

The target date has to exist in `nvidea_stocks_pred.csv` and have at least 50 rows of history before it — otherwise it raises `KeyError` or `ValueError` rather than guessing. Right now the target date is hardcoded (`2026-05-20`, which falls inside the CSV's `2026-01-30` to `2026-05-29` range) as a default argument in `predict_tomorrow_open()`; the CLI doesn't currently prompt for a different one, so you'd need to call the function directly or edit the default to check another date.

## Getting started

```bash
uv sync
uv run main.py
```

You'll be prompted:
- `t` — trains a new model on `nvidea_stocks.csv` and overwrites `model/model.pt`.
- `p` — runs a prediction for the hardcoded target date and prints the estimated price.
- anything else — exits.

## Disclaimer

This is a portfolio/learning project, not investment advice. A small LSTM over three engineered features isn't a serious market predictor — stock prices are driven by far more than the last 50 days of volatility, returns, and volume. Don't trade real money based on this.
