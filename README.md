# Islamabad Housing Price Prediction

Machine learning pipeline for predicting housing price ranges in Islamabad using real estate listings from [Zameen.com].

## Overview

This project uses cleaned and engineered housing data to classify property prices into 5 quantile-based price categories.

The pipeline includes:

* Data cleaning and preprocessing
* Geographic filtering for Islamabad-only listings
* Low-frequency area handling
* Feature engineering
* Stratified geographic train/test splitting
* Classification using XGBoost
* FastAPI deployment for local inference

Regression models were also explored in the research notebook and will be integrated into the production pipeline later.

---

## Dataset Processing

Key preprocessing steps:

* Removed listings outside Islamabad
* Handled missing and inconsistent values
* Removed low-frequency locations
* Encoded categorical features
* Applied geographic balancing strategy using:

```python
run_macro_stratified_split(df)
```

This custom split strategy improves regional balance across evaluation folds.

---

## Model

### Classification

* Model: `XGBClassifier`
* Target: 5 quantile-based housing price classes

### Regression

Regression experiments are available in the notebook but are not yet integrated into the `src/` pipeline.

---

## FastAPI Deployment

A local FastAPI server is included for real-time predictions.

Users can input housing features and receive predicted price categories through the API docs interface.

---

## Usage

### Train the Model

```bash
python main.py
```

### Run FastAPI Server

```bash
python app.py
```

---

## Current Status

* Classification pipeline implemented
* FastAPI deployment working locally
* Regression pipeline pending integration
* Metrics and evaluation reports will be added soon

---

## Tech Stack

* Python
* Pandas
* Scikit-learn
* XGBoost
* FastAPI
* Jupyter Notebook
