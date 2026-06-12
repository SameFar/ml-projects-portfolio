import pandas as pd
from pathlib import Path

def get_data():
    # from https://www.kaggle.com/datasets/suryaprabha19/fashion-and-color-recommendation-dataset
    data_path = Path(__file__).resolve().parent.parent.parent / 'data' / 'recommendations.csv'
    
    dfc = pd.read_csv(data_path)
    
    X = pd.get_dummies(dfc[['Hair Color', 'Eye Color',
                        'Skin Tone', 'Under Tone',
                        'Torso length']], dtype=float, drop_first=True).values
    
    y = pd.get_dummies(dfc[['Recommended Clothing Colors', 'Avoid Clothing Colors',
       'Recommended Materials', 'Recommended Patterns', 'Recommended Jewelry Metal', 
       'Recommended Clothing Color Wheel Region']], dtype=float).values

    return X, y

