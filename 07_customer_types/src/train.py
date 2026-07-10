from sklearn.mixture import GaussianMixture
from pathlib import Path
from data_preprocessing import get_clean_data
import pickle
import pandas as pd


def train():
    # Dataset from https://archive.ics.uci.edu/dataset/352/online+retail
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / "data" / "online_retail_II.xlsx"
    gmm = GaussianMixture(n_components=6)

    initial_df = pd.read_excel(DATA_DIR).dropna()
    df = get_clean_data(initial_df)

    gmm.fit(df)

    with open(BASE_DIR / "model" / "model.pkl", "wb") as m:
        pickle.dump(gmm, m)

    print("Model trained")


if __name__ == "__main__":
    train()
