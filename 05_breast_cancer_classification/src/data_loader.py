import os
import pandas as pd

def load_cancer_dataset(filepath=None):
    if filepath is None:
        filepath = "/home/sameer/Documents/jupyter/ml-projects-portfolio/data/breastcancer.csv"
        
    df = pd.read_csv(filepath).drop_duplicates()
    
    # Pruning features showing close-to-zero correlation bounds (|r| < 0.01) with the target
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

def process_and_split_features(df):
    # Safe categorical extraction to binary map: Malignant = 1, Benign = 0
    dummies = pd.get_dummies(df['diagnosis'], dtype=int)
    df_processed = pd.concat([df, dummies], axis=1)
    
    # Dropping the intercept category (B) alongside metadata keys to isolate clean numerical fields
    X = df_processed.drop(columns=["id", "B", "diagnosis", "M"], errors="ignore")
    y = df_processed["M"]
    
    return X, y