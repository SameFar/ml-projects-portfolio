import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_and_clean_data(file_path):
    """
    Executes regional token cleaning, spatial metric normalization (Marlas),
    and sets up discrete target classification brackets.
    """
    # dataset from https://www.kaggle.com/datasets/diraf0/pakistan-housing-dataset
    raw_df = pd.read_csv(file_path).drop_duplicates().dropna()

    # Restrict scope to Islamabad records and strip unresolved range strings
    isl_df = raw_df[
        raw_df["Location"].str.contains("Islamabad", case=False, na=False)
    ].copy()
    isl_df = isl_df[
        ~isl_df["Baths"].str.contains("-", na=False)
        & ~isl_df["Beds"].str.contains("-", na=False)
    ]

    isl_df["Baths"] = isl_df["Baths"].astype(int)
    isl_df["Beds"] = isl_df["Beds"].astype(int)

    # Filter out low-frequency sectors to eliminate sparse column noise
    clean_df = isl_df[
        isl_df.groupby("Location")["Location"].transform("count") >= 25
    ].copy()

    # Clean location strings and compile dummy variables
    clean_df["Location"] = clean_df["Location"].str.replace(
        ", Islamabad, Islamabad Capital", "", regex=False
    )

    # Concat dummy variables and drop reference sector 'B-17' to avoid multi-collinearity
    dummies = pd.get_dummies(clean_df["Location"], dtype=int)
    clean_df = pd.concat([clean_df, dummies], axis=1).drop(columns=["Location", "B-17"])

    # Regional Area Conversion: Standardize Kanals down to base Marlas (1 Kanal = 20 Marlas)
    area_parts = clean_df["Area"].str.lower().str.split(expand=True)
    area_vals = pd.to_numeric(area_parts[0])
    area_units = area_parts[1]
    clean_df["Marla"] = area_vals.where(area_units == "marla", area_vals * 20.0).astype(
        float
    )
    clean_df.drop(columns=["Area"], inplace=True)

    # Regional Currency Conversion: Parse Lakh/Crore/Arab strings into raw Crore floats
    price_parts = clean_df["Price"].str.split(expand=True)
    price_vals = price_parts[0].str.replace("PKR", "", regex=False).astype(float)
    price_units = price_parts[1]

    unit_multiplier = price_units.map({"Crore": 1.0, "Lakh": 0.01, "Arab": 100.0})
    clean_df["Crore"] = (price_vals * unit_multiplier).astype(float)
    clean_df.drop(columns=["Price"], inplace=True)

    # Secondary Structural Spaces Aggregation
    room_features = [
        "Steam Room",
        "Prayer Rooms",
        "Dining Room",
        "Laundry Room",
        "Drawing Room",
        "Lounge or Sitting Room",
        "Powder Room",
    ]
    clean_df["Other Rooms"] = clean_df[room_features].sum(axis=1)

    dropped_cols = room_features + [
        "Store Rooms",
        "Kitchens",
        "Gym",
        "No additional rooms",
    ]
    clean_df.drop(columns=dropped_cols, inplace=True)

    # Discretize continuous pricing curves into target ordinal brackets
    bins = [0, 5, 10, 15, 20, np.inf]
    labels = [0, 1, 2, 3, 4]
    clean_df["price_category"] = pd.cut(
        clean_df["Crore"], bins=bins, labels=labels, right=False
    ).astype(int)

    return clean_df


def run_macro_stratified_split(df):
    """
    Enforces geographic balance across evaluation folds via a regional proxy matrix.
    """
    X = df.drop(columns=["Crore", "price_category"])
    y = df["price_category"]

    def tag_macro_neighborhoods(row):
        if row.get("Bahria Town", 0) == 1:
            return "Bahria"
        if row.get("G-13", 0) == 1:
            return "G13"
        if row.get("DHA Defence", 0) == 1:
            return "DHA"
        return "Other"

    stratify_proxy = X.apply(tag_macro_neighborhoods, axis=1)
    return train_test_split(
        X, y, test_size=0.2, stratify=stratify_proxy, random_state=42
    )
