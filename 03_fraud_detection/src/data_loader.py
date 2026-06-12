import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Download dataset from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# Edit DEFAULT_PATH
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_PATH = BASE_DIR.parent / "data" / "creditcard.csv"

def load_and_engineer_data(data_path=DEFAULT_PATH):
    """
    Loads credit card entries, applies cyclical time transformations,
    scales heavy monetary skews, and strips overlapping feature channels.
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Missing creditcard.csv at expected destination: {data_path}")
        
    df = pd.read_csv(data_path).drop_duplicates().dropna()
    
    # 1. Cyclical Time Transformations
    df["Hour"] = (df["Time"] // 3600) % 24
    df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
    
    # 2. Skewed Feature Log-scaling
    df['LogAmount'] = np.log1p(df['Amount'])
    
    # 3. Target Extraction & Feature Selection based on KDE convergence
    y = df["Class"].values
    
    drop_cols = ["Class", "Time", "Amount", "Hour", "V13", "V15", "V20", "V22", "V23", "V25", "V28"]
    X = df.drop(columns=drop_cols).values
    
    return train_test_split(X, y, test_size=0.2, random_state=42)