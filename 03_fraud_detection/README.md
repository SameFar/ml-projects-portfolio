# Project Name: Imbalanced Anomaly Detection: Credit Card Fraud via XGBoost

A production-grade classification pipeline utilizing feature engineering and optimized gradient boosting to securely identify fraudulent credit card transactions.

## 💡 The Quirk & Learning Objective
This milestone shifts focus from standard datasets to **the challenges of extreme class imbalance (0.17% target frequency)**. 

To solve this, I designed a multi-step pipeline focused on optimization, signal processing, and target metrics:
* **Cyclical Temporal Mapping:** Transformed linear transaction timestamps into standard harmonic dimensions using sine/cosine frequency modulations:
  $$\text{Hour}_{\text{sin}} = \sin\left(\frac{2\pi \cdot \text{Hour}}{24}\right), \quad \text{Hour}_{\text{cos}} = \cos\left(\frac{2\pi \cdot \text{Hour}}{24}\right)$$
  This teaches tree ensembles that hour 23 and hour 00 are temporally adjacent.
* **KDE Divergence Pruning:** Generated full-scale Kernel Density Estimate comparisons for all variables. Features where the fraudulent and valid distributions converged completely (e.g., `V13`, `V15`) were dropped as zero-variance noise to streamline calculation and prevent overfitting.
* **False Negative Risk Minimization:** Engineered a multi-metric custom `GridSearchCV` harness. While tracking both precision and recall, the pipeline explicitly refits hyperparameter matrices based on **Recall optimization**, ensuring the model prioritizes identifying actual anomalies.

## 📁 Repository Structure
* `portfolio_notebook.ipynb`: Exploratory notebook containing full data distribution visualizations, density overlap analysis, and initial metric tracking.
* `src/`: Clean, production-ready modules.
  * `data_loader.py`: Processes file pathways, applies cyclical maps, and strips uninformative features.
  * `model.py`: Instantiates scaling pipelines and grid-search wrappers.
  * `train.py`: Coordinates the workflow, logs exact matrix outcomes, and generates visualization charts.
* `results/`: Performance documentation directory.
  * `results.txt`: Text logs detailing dataset counts, explicit precision records, and targeted recall achievements.
  * `confusion_matrix.png`: Annotated validation matrix tracking true positives versus missed alerts.

## 📊 Performance Records & Analysis
The baseline model achieves excellent classification control over the unbalanced testing set:
* **Precision:** `94.44%` (Minimizing false positives that disrupt clean user transactions)
* **Recall:** `75.56%` (Successfully capturing the majority of anomalous attacks)

For full parameter settings and a detailed look at the true/false count distributions, refer to `results/results.txt`.

## 🚀 Execution & Portability
This repository uses entirely relative path configurations. It will run out of the box on any device without modifying directory locations.

Install dependencies:
```bash
pip install -r requirements.txt