import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

def load_clean_split_data(test_size=0.2, random_state=42):
    """
    Fetches California housing data, scrubs outliers using the IQR method,
    logs the target feature, and returns split datasets.
    """
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    
    # 1. Outlier removal using IQR
    cols = df.drop(columns="MedHouseVal").columns
    Q1 = df[cols].quantile(0.25)
    Q3 = df[cols].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    mask = (df[cols] >= lower_bound) & (df[cols] <= upper_bound)
    df_clean = df[mask.all(axis=1)]
    
    # 2. Separate features and target
    X = df_clean.drop(columns="MedHouseVal").values
    y = df_clean["MedHouseVal"].values
    
    # 3. Target transformation
    y_logged = np.log1p(y)
    
    return train_test_split(X, y_logged, test_size=test_size, random_state=random_state)