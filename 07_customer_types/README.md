<div align="center">

# 🛒 Customer Types — E-Commerce Segmentation

**Gaussian Mixture clustering of shoppers, served behind FastAPI.**

![Machine Learning](https://img.shields.io/badge/🔵_MACHINE_LEARNING-1f6feb?style=for-the-badge)

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)

</div>

---

Groups e-commerce customers into behavioral segments from raw transaction data using a Gaussian Mixture Model. A FastAPI endpoint takes a customer's yearly stats and returns their segment.

## 🧠 How it works

The source is the [UCI Online Retail II](https://archive.ics.uci.edu/dataset/352/online+retail) dataset — one row per invoice line item. `src/data_preprocessing.py` groups by `Customer ID` so each row is one customer, engineering seven features:

- **Transactions** — line items bought
- **Total Quantity** — total units purchased
- **Total Spent** — price × quantity, summed
- **Total Refund Received** — money back from returns (negative quantities)
- **Avg Spent** — total spent ÷ transaction count
- **Purchases per month** — transactions ÷ 12 (the dataset spans ~a year)
- **Return Rate** — transaction count relative to refund count

Features are scaled with `RobustScaler`, since a few very high spenders would dominate a standard scaler. The exploration notebook tried KMeans, HDBSCAN, and GMM side by side in 3D; GMM gave the cleanest separation, so `src/train.py` fits a 6-component GMM saved to `model/model.pkl`.

`app.py` maps the 6 cluster indices to human-readable labels from manual inspection of the cluster means:

| Cluster | Label |
| --- | --- |
| 0 | No Refunds |
| 1 | Significant Refund rate |
| 2 | Frequent Buyers |
| 3 | Anomaly |
| 4 | Big Spenders – Infrequent |
| 5 | Anomaly |

Clusters 3 and 5 came out as outlier groups rather than distinct behaviors, so both are labeled "Anomaly." `src/pydantic_model.py` only asks the caller for raw counts and derives `Avg_Spent`, `Transactions_per_month`, and `Return_Rate` as Pydantic computed fields.

## 📊 Results

This is unsupervised, so there's no accuracy/F1 — GMM was chosen over KMeans/HDBSCAN qualitatively, on visual cluster separation in the 3D scatter plots.

## 🚀 Getting started

```bash
uv sync
uv run -m src.train  # fits the GMM on data/online_retail_II.xlsx, saves model/model.pkl
uv run app.py        # starts the FastAPI server at http://127.0.0.1:8000
```

`POST /predict` takes `Transactions`, `Quantity`, `Spent`, `Refunds`, and `Returns`, and returns the predicted segment label.
