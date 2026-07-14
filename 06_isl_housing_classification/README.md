<div align="center">

# 🏘️ Islamabad Housing Price Classification

**Price-bracket prediction on scraped listings, with a FastAPI wrapper.**

![Machine Learning](https://img.shields.io/badge/🔵_MACHINE_LEARNING-1f6feb?style=for-the-badge)

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-337AB7?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)

</div>

---

Predicts which price bracket an Islamabad property falls into, from listings scraped off Zameen.com ([Pakistan housing dataset](https://www.kaggle.com/datasets/diraf0/pakistan-housing-dataset)). Ships an XGBoost model plus a small FastAPI wrapper for local inference.

## 🧠 How it works

The raw dataset spans all of Pakistan, so `src/data_preprocessing.py` filters to Islamabad rows, drops malformed bed/bath counts (ranges like "3-4"), and removes sectors with fewer than 25 listings so the one-hot location columns aren't noise. It also:

- converts area to a single **Marla** unit (Kanals multiplied out),
- parses price strings (`PKR3.5 Crore`, `PKR50 Lakh`…) into a Crore-denominated float,
- collapses rare room-type columns (steam / prayer / drawing room…) into one `Other Rooms` count,
- buckets price into 5 brackets: under 5 Crore, 5–10, 10–15, 15–20, and 20+.

The split isn't plain random — `run_macro_stratified_split` tags each row by whether it's in Bahria Town, G-13, or DHA Defence (the three largest neighborhoods) and stratifies on that, so the big neighborhoods are represented proportionally in both sets.

`src/models.py` defines an `XGBClassifier` pipeline that `main.py` trains and pickles to `saved_models/housing_xgb_pipeline.pkl`, along with the exact feature columns it was trained on. (A regression variant — predicting actual price, not a bracket — was explored in `Islamabad Housing-Regression.ipynb`, notebook-only.)

## 📊 Results

`main.py` on the held-out 20% split:

- **Accuracy: 0.76**
- Best on bracket 4 (20+ Crore, F1 0.91) and bracket 0 (under 5 Crore, F1 0.89) — the cheap and expensive extremes are easiest to call.
- Weakest on bracket 2 (10–15 Crore, F1 0.32) and bracket 3 (15–20 Crore, F1 0.50) — the middle brackets are the smallest classes and get confused with their neighbors.

## 🚀 Getting started

```bash
uv sync
uv run main.py       # trains the XGBoost pipeline, prints accuracy/classification report, saves to saved_models/
uv run app.py            # starts the FastAPI server at http://127.0.0.1:8000
```

The API needs `saved_models/housing_xgb_pipeline.pkl`, so train first. `POST /predict` takes `baths`, `beds`, `marla`, `rooms`, and `location` (a sector like `"G-13"`) and returns the predicted price bracket.
