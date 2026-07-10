# Breast Cancer Classification — 8-Model Consensus Ensemble

Classifies breast tumor biopsies as malignant or benign using the classic [Wisconsin breast cancer dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data). Instead of picking one model, this trains eight different classifiers and combines their votes into a single conservative decision.

## How it works

The data is 569 biopsy samples with 30 measured features (radius, texture, smoothness, etc.), each computed as mean/standard-error/worst across cell nuclei in the sample. Five features with very low correlation to the diagnosis (`|r| < 0.01`, things like `texture_se` and `fractal_dimension_mean`) get dropped, leaving 25 features.

The eight models split into two groups that get different preprocessing:

- **Distance/gradient-based models** (Logistic Regression, SGD, SVC, KNN, Naive Bayes): fit on data that's been passed through a Yeo-Johnson power transform (to reduce skew) and had outliers stripped out with an `IsolationForest` (7% contamination) beforehand. These models care about scale and are sensitive to leverage points, so they get the cleanup.
- **Tree-based models** (Random Forest, XGBoost, Decision Tree): fit on the raw, untransformed data, since trees split on thresholds and don't care about scale or a handful of outliers.

`src/main.py` runs an 80/20 stratified train/test split, trains all eight, scores each on accuracy/precision/recall/F1, writes the comparison table to `results/results.txt`, and pickles every trained model into `saved_models/`.

`src/predict.py` loads all eight saved models and runs a sample through every one of them. Instead of majority vote, the final call is a conservative threshold: if 4 or more of the 8 models flag "malignant," the case is called malignant. The idea is to bias the whole system toward catching more true positives at the cost of some false alarms, which is generally what you'd want in a screening context.

## Results

From `results/results.txt` (test set, 20% holdout):

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

The three models on Yeo-Johnson-transformed, outlier-cleaned data (Logistic Regression, SGD, SVC) came out on top — once the skew is stabilized, the decision boundary between malignant and benign is mostly linear, so the simpler linear/margin-based models edge out the tree ensembles here.

## Getting started

```bash
uv sync
uv run src/main.py       # trains all 8 models, writes results/results.txt and saved_models/*.pkl
uv run src/predict.py    # loads the saved models and runs the ensemble vote on data/sample.csv
```

`src/predict.py` expects `data/sample.csv` to contain rows with the 25 features listed in `EXPECTED_FEATURES` in that file — it picks a random row from that CSV and prints each model's vote plus the final consensus call.
