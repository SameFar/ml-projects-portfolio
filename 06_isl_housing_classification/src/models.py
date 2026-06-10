from sklearn.pipeline import Pipeline
from sklearn.preprocessing import  StandardScaler
from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor, XGBClassifier

def get_pipelines():
    return {
        'XGBClass' : Pipeline([
    ('scale', StandardScaler()), 
    ('model', XGBClassifier(
        learning_rate=0.3,
        max_depth=3,
        min_child_weight=0,
        reg_alpha=0.05,
        reg_lambda=1.5,
        use_label_encoder=False,
        eval_metric='mlogloss'
    ))
]),
        'XGBReg' : Pipeline([
    ('scale', StandardScaler()), 
    ('model', XGBRegressor(
        learning_rate=0.3,
        max_depth=3,
        min_child_weight=0,
        reg_alpha=0.05,
        reg_lambda=1.5,
        use_label_encoder=False,
        eval_metric='rmse'
    ))
]),
'RandomForest' : Pipeline([
    ('scale', StandardScaler()), 
    ('model', RandomForestRegressor(
        n_estimators= 300,
         min_samples_split= 5,
         min_samples_leaf= 1,
         max_features= 1.0,
         max_depth= 30,
         bootstrap = True
       )
    )
])
}
