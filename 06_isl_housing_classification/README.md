# Islamabad Housing Price Classification

Predicts which price bracket an Islamabad property listing falls into, using real estate listings scraped from Zameen.com ([Pakistan housing dataset on Kaggle](https://www.kaggle.com/datasets/diraf0/pakistan-housing-dataset)). Ships with an XGBoost model and a small FastAPI wrapper for local inference.

## How it works

The raw dataset covers listings from all over Pakistan, so `src/data_preprocessing.py` first filters down to Islamabad-only rows, drops listings with malformed bed/bath counts (ranges like "3-4"), and removes sectors with fewer than 25 listings so the one-hot location columns aren't mostly noise. It also:

- converts area into a single "Marla" unit (Kanals get multiplied out to their Marla equivalent),
- parses the price strings (`PKR3.5 Crore`, `PKR50 Lakh`, etc.) into a single Crore-denominated float,
- collapses a handful of rarely-used room-type columns (steam room, prayer room, drawing room, ...) into one `Other Rooms` count,
- and buckets price into 5 fixed brackets: under 5 Crore, 5-10, 10-15, 15-20, and 20+.

The train/test split isn't a plain random split — `run_macro_stratified_split` tags each row by whether it's in Bahria Town, G-13, or DHA Defence (the three largest neighborhoods) and stratifies on that, so the big neighborhoods are represented proportionally in both sets.

`src/models.py` defines an `XGBClassifier` pipeline (with `StandardScaler`, though XGBoost doesn't strictly need it) that `src/main.py` trains and pickles to `saved_models/housing_xgb_pipeline.pkl`, along with the exact list of feature columns it was trained on.

There's also a separate `Islamabad Housing-Regression.ipynb` notebook where a regression version of this (predicting an actual price, not a bracket) was explored, with its own FastAPI experiment. That's notebook-only — it was never turned into a `src/` module, so it's not part of the pipeline you'd actually run.

## Results

Running `src/main.py` on the held-out 20% split gives:

- **Accuracy: 0.76**
- Best per-class performance on bracket 4 (20+ Crore, F1 0.91) and bracket 0 (under 5 Crore, F1 0.89) — the cheap and expensive extremes are easiest to call.
- Weakest on bracket 2 (10-15 Crore, F1 0.32) and bracket 3 (15-20 Crore, F1 0.50) — the middle brackets are the smallest classes and get confused with their neighbors.

## Getting started

```bash
uv sync
uv run src/main.py       # trains the XGBoost pipeline, prints accuracy/classification report, saves to saved_models/
cd src && uv run app.py  # starts the FastAPI server at http://127.0.0.1:8000
```

The API expects `saved_models/housing_xgb_pipeline.pkl` to exist, so run training first. `POST /predict` takes `baths`, `beds`, `marla`, `rooms`, and `location` (a sector name like `"G-13"`) and returns the predicted price bracket.
