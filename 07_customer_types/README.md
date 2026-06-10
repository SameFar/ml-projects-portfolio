# E-Commerce Customer Segmentation

A machine learning pipeline that groups e-commerce customers into different segments based on their buying habits. The project uses a Gaussian Mixture Model (GMM) to group customers and a FastAPI server to handle real-time predictions.

## Overview

This project takes raw sales data and turns it into useful profiles by grouping customer behaviors.

The pipeline includes:

* Data cleaning and dropping missing values
* Feature engineering to group transactions by unique customers
* Handling refunds and returns mathematically
* Scaling features to handle big spenders safely
* Clustering customers using a Gaussian Mixture Model (GMM)
* FastAPI deployment to predict a customer's segment locally

---

## Dataset & Feature Engineering

The raw data comes from a retail sales spreadsheet. The pipeline cleans the data and groups it by `Customer ID` so that each row represents one customer instead of a single receipt line.

Calculated features per customer:

* **Transactions:** Total number of times they bought something
* **Total Quantity:** Total number of items purchased
* **Total Spent:** Total money spent (Price $\times$ Quantity)
* **Total Refund Received:** Total money refunded from returned items
* **Avg Spent:** Average money spent per transaction
* **Purchases per month:** Average transactions made per month
* **Return Rate:** How often they return items compared to buying

---

## Model

A **Gaussian Mixture Model (GMM)** with 6 clusters was chosen because it handles overlapping customer profiles better than standard K-Means. Features are scaled using `RobustScaler` to make sure extreme shoppers don't break the model.

### Customer Segments Created:

* `0`: Standard Buyers (No Refunds)
* `1`: High-Volume Accounts (High Refund Rate)
* `2`: Frequent / Loyal Buyers
* `3`: Outlier Group A
* `4`: Big Spenders (Buy Infrequently)
* `5`: Outlier Group B

---

## FastAPI Deployment

A local FastAPI server handles real-time predictions.

It uses Pydantic to validate input data and automatically calculates missing fields (like average spent and monthly purchase rates) on the fly before running the model.

---

## Usage

### Train the Model

```bash
python train.py

```

### Run FastAPI Server

```bash
python app.py

```

---

## Current Status

* Customer data grouping logic complete
* GMM model trained and saved as `model.pkl`
* FastAPI server working locally with auto-calculated fields
* Cluster visual plots pending integration into the app

---

## Tech Stack

* Python
* Pandas
* Scikit-learn
* FastAPI
* Pydantic
* Uvicorn

---