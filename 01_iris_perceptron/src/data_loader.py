import pandas as pd

def load_and_preprocess(data_path):
    """
    Loads iris dataset, cleans nulls/duplicates, and isolates 
    virginica for binary classification.
    """
    df = pd.read_csv(data_path).drop_duplicates().dropna()
    
    # OneHot encode and isolate virginica target by dropping other classes
    df = pd.concat([df, pd.get_dummies(df["species"])], axis=1)
    df = df.drop(columns=["species", "setosa", "versicolor"])
    
    X_data = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
    y_data = df['virginica'].values
    
    return df, X_data, y_data