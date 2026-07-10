from pathlib import Path
import pandas as pd
from sklearn.preprocessing import RobustScaler


def get_clean_data(initial_df):

    scale = RobustScaler()

    clean_df = initial_df.drop(columns=["Invoice", "Description"])

    # Calculate total money spent per transaction
    clean_df["Total"] = clean_df["Price"] * clean_df["Quantity"]

    mask = clean_df["Quantity"] < 0
    refunds = clean_df.loc[mask]

    customers = clean_df.groupby("Customer ID")

    # Make new df for customer data
    df = pd.DataFrame()

    df["Transactions"] = customers["Customer ID"].count()
    df["Total Quantity"] = customers["Quantity"].sum()
    df["Total Spent"] = (customers["Total"].sum()).round(2)
    df["Total Refund Recived"] = abs(refunds.groupby("Customer ID")["Total"].sum())
    df["Total Refund Recived"] = df["Total Refund Recived"].fillna(0)
    df["Avg Spent"] = (df["Total Spent"] / df["Transactions"]).round(2)
    df["Purchases per month"] = (df["Transactions"] / 12).round(2)
    df["Return Rate"] = (
        customers["Customer ID"].count()
        // refunds.groupby("Customer ID")["Total"].count()
    )
    df["Return Rate"] = df["Return Rate"].fillna(0)

    df.to_csv(Path(__file__).resolve().parent.parent / "data" / "customer_summary.csv")

    return scale.fit_transform(df)
