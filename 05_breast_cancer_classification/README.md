<div align="center">

# 🔬 Breast Cancer Classification

**Eight classifiers, one conservative consensus vote.**

![Machine Learning](https://img.shields.io/badge/🔵_MACHINE_LEARNING-1f6feb?style=for-the-badge)

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-337AB7?style=flat-square)

</div>

---

Classifies breast tumor biopsies as malignant or benign on the [Wisconsin breast cancer dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data). Instead of one model, it trains eight and combines their votes into a single conservative decision.

## 🧠 How it works

569 biopsy samples, 30 features each (radius, texture, smoothness…) computed as mean / standard-error / worst across cell nuclei. Five features with near-zero correlation to the diagnosis (`|r| < 0.01`) are dropped, leaving 25.

The eight models split into two groups with different preprocessing:

- **Distance / gradient-based** (Logistic Regression, SGD, SVC, KNN, Naive Bayes) — fit on data passed through a Yeo-Johnson power transform (reduces skew) with outliers stripped by an `IsolationForest` (7% contamination). These care about scale and leverage points, so they get the cleanup.
- **Tree-based** (Random Forest, XGBoost, Decision Tree) — fit on raw data, since trees split on thresholds and ignore scale and a few outliers.

`main.py` runs an 80/20 stratified split, trains all eight, scores accuracy / precision / recall / F1, writes the table to `results/results.txt`, and pickles each model into `saved_models/`.

`src/predict.py` loads all eight and votes: if **4 or more** flag malignant, the case is called malignant — deliberately biased toward catching true positives, as you'd want in a screening context.

## 📊 Results

From `results/results.txt` (20% holdout):

| Model | Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| Logistic Regression | 0.9912 | 0.9913 | 0.9912 | 0.9912 |
| SGD | 0.9912 | 0.9913 | 0.9912 | 0.9912 |
| SVC | 0.9912 | 0.9913 | 0.9912 | 0.9912 |
| Naive Bayes | 0.9825 | 0.9825 | 0.9825 | 0.9825 |
| XGBoost | 0.9649 | 0.9668 | 0.9649 | 0.9645 |
| Random Forest | 0.9649 | 0.9668 | 0.9649 | 0.9645 |
| KNN | 0.9561 | 0.9569 | 0.9561 | 0.9558 |
| Decision Tree | 0.9298 | 0.9297 | 0.9298 | 0.9294 |

The three models on Yeo-Johnson-transformed, outlier-cleaned data came out on top — once skew is stabilized, the malignant/benign boundary is mostly linear, so the simpler linear/margin models edge out the tree ensembles.

## 🚀 Getting started

```bash
uv sync
uv run main.py       # trains all 8 models, writes results/results.txt and saved_models/*.pkl
uv run -m src.predict    # loads the saved models and runs the ensemble vote on data/sample.csv
```

`src/predict.py` expects `data/sample.csv` with the 25 features listed in `EXPECTED_FEATURES` — it picks a random row and prints each model's vote plus the final consensus call.
