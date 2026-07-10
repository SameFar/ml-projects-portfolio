import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


def clean_training_outliers(X_train, y_train, contamination=0.02):
    # Fit strictly on train split to prevent data/target leakage
    iso = IsolationForest(contamination=contamination, random_state=42)
    inliers = iso.fit_predict(X_train)

    return X_train[inliers == 1], y_train[inliers == 1]


def get_baseline_models():
    xgb = XGBRegressor(
        learning_rate=0.03,
        max_depth=6,
        n_estimators=200,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_weight=1,
        random_state=42,
    )

    rf = RandomForestRegressor(
        n_estimators=500,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        max_features="log2",
    )

    # SVR requires explicit scaling due to distance sensitivity
    svr = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", SVR(kernel="rbf", C=2, gamma=0.02, epsilon=0.02)),
        ]
    )

    return {
        "XGBoost Regressor": xgb,
        "Random Forest Regressor": rf,
        "Support Vector Regressor": svr,
    }


def evaluate_predictions(y_true_log, y_pred_log):
    # Reverse the log1p transform applied during the preprocessing phase
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)

    metrics = {
        "R2": round(r2_score(y_true, y_pred), 3),
        "MAE": round(mean_absolute_error(y_true, y_pred), 2),
        "MSE": round(mean_squared_error(y_true, y_pred), 2),
    }
    return metrics
