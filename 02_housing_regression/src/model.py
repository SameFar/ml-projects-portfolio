from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer
from sklearn.ensemble import HistGradientBoostingRegressor

def get_regression_pipeline():
    """
    Constructs a machine learning pipeline that scales skewed distributions
    using a QuantileTransformer prior to fitting a HistGradientBoostingRegressor.
    """
    pipeline = Pipeline([
        (
            "quantile", 
            QuantileTransformer(output_distribution="normal", random_state=42)
        ),
        (
            "model", 
            HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_depth=6,
                max_iter=800,
                min_samples_leaf=10,
                l2_regularization=0.1,
                random_state=42
            )
        )
    ])
    return pipeline