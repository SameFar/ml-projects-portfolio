import pandas as pd

def load_cancer_dataset(filepath) -> pd.DataFrame:
    # Dataset from https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data
        
    df = pd.read_csv(filepath).drop_duplicates()
    
    # Removing features showing very low correlation (|r| < 0.01) with y
    uninformative_features = [
        "texture_se", 
        "smoothness_se", 
        "symmetry_se", 
        "fractal_dimension_se", 
        "fractal_dimension_mean"
    ]
    df = df.drop(columns=uninformative_features, errors="ignore")
    
    return df

import pandas as pd

def process_and_split_features(df: pd.DataFrame) -> pd.DataFrame:
    # Encoded: Malignant = 1, Benign = 0
    dummies = pd.get_dummies(df['diagnosis'], dtype=int)
    df_processed = pd.concat([df, dummies], axis=1)
    
    # Dropping the intercept category (B) and metadata keys
    X = df_processed.drop(columns=["id", "B", "diagnosis", "M"], errors="ignore")
    y = df_processed["M"]
    
    return X, y