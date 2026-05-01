import pandas as pd
if __name__ == "__main__":
    df = pd.read_csv("data/processed/real_dataset.csv")
    print("Total samples:", len(df))
    print(df["label"].value_counts())
    print("Class ratio (label=0 vs 1):")
    print(df["label"].value_counts(normalize=True))
