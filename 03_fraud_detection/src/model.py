from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, make_scorer
from xgboost import XGBClassifier

def get_baseline_pipeline():
    """
    Returns an un-tuned production scaling and classification pipeline.
    """
    return Pipeline([
        ("scale", RobustScaler()),
        ("model", XGBClassifier(
            learning_rate=0.15, 
            max_depth=7, 
            n_estimators=1000, 
            random_state=42,
            eval_metric="logloss"
        ))
    ])

def get_grid_search(pipeline):
    """
    Wraps an existing pipeline configuration into a target-focused 
    multi-metric GridSearch optimizer.
    """
    param_grid = {
        "model__n_estimators": [300, 600, 1000],
        "model__max_depth": [3, 5, 7],
        "model__learning_rate": [0.01, 0.05, 0.1],
    }
    
    # Track both metrics, but refit the final model based strictly on Recall performance
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring={
            "precision": make_scorer(precision_score), 
            "recall": make_scorer(recall_score)
        },
        refit="recall",
        cv=4,
        n_jobs=-1,
        verbose=1
    )
    return grid