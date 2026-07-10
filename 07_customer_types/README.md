# Customer Types — E-Commerce Segmentation

Groups e-commerce customers into behavioral segments from raw transaction data, using a Gaussian Mixture Model. Comes with a FastAPI endpoint that takes a customer's yearly stats and returns which segment they fall into.

## How it works

The source data is the [UCI Online Retail II](https://archive.ics.uci.edu/dataset/352/online+retail) dataset — one row per line item on an invoice. `src/data_preprocessing.py` groups everything by `Customer ID` so each row becomes one customer, and engineers seven features per customer:

- **Transactions** — how many line items they've bought
- **Total Quantity** — total units purchased
- **Total Spent** — price times quantity, summed
- **Total Refund Received** — money back from returns (quantities under 0 are treated as refunds)
- **Avg Spent** — total spent divided by transaction count
- **Purchases per month** — transactions divided by 12 (the dataset spans roughly a year)
- **Return Rate** — transaction count relative to refund count

Features are scaled with `RobustScaler` before clustering, since a handful of very high spenders would otherwise dominate a standard scaler.

The exploration notebook (`Retail Customer Clustering.ipynb`) tried KMeans, HDBSCAN, and a Gaussian Mixture Model side by side, plotting each in 3D. GMM gave the cleanest separation, so that's what `src/train.py` actually fits — 6 components, saved to `model/model.pkl`.

The FastAPI app (`src/app.py`) maps each of the 6 cluster indices to a human-readable label based on manual inspection of the cluster means:

| Cluster | Label |
| --- | --- |
| 0 | No Refunds |
| 1 | Significant Refund rate |
| 2 | Frequent Buyers |
| 3 | Anomaly |
| 4 | Big Spenders - Infrequent |
| 5 | Anomaly |

Two of the six clusters (3 and 5) came out looking like outlier groups rather than distinct behavioral types, so both just get labeled "Anomaly."

`src/pydantic_model.py` defines the API's request schema — it only asks the caller for the raw counts (transactions, quantity, spent, refunds, returns) and computes `Avg_Spent`, `Transactions_per_month`, and `Return_Rate` itself as Pydantic computed fields, so the caller doesn't have to derive them.

## Results

This is unsupervised clustering, so there's no accuracy/F1 to report — the notebook's justification for picking GMM over KMeans/HDBSCAN is qualitative (visual cluster separation in the 3D scatter plots), not a benchmarked metric.

## Getting started

```bash
uv sync
cd src && uv run train.py  # fits the GMM on data/online_retail_II.xlsx, saves model/model.pkl
cd src && uv run app.py    # starts the FastAPI server at http://127.0.0.1:8000
```

`POST /predict` takes `Transactions`, `Quantity`, `Spent`, `Refunds`, and `Returns`, and returns the predicted segment label.
