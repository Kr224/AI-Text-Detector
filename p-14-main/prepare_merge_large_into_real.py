# merge_large_into_real.py
"""
Merge the synthetic large_dataset.csv into real_dataset.csv.

After running this script, the file:
    data/processed/real_dataset.csv

will contain both:
    - existing real dataset rows (if any)
    - all rows from large_dataset.csv

Expected schema for both CSV files:
    text,label
    <string>,0 or 1

Where:
    label = 0 -> human-written text
    label = 1 -> AI-generated text
"""

import os
import pandas as pd


def merge_large_into_real(
    real_path: str = "data/processed/real_dataset.csv",
    large_path: str = "data/processed/large_dataset.csv",
    drop_duplicates: bool = True,
    shuffle: bool = True,
    random_state: int = 42,
):
    """Merge large_dataset into real_dataset and overwrite real_dataset.csv."""
    # ----- Load real_dataset if it exists -----
    if os.path.exists(real_path):
        print(f"[INFO] Loading existing real dataset from: {real_path}")
        df_real = pd.read_csv(real_path)
        # Basic sanity check
        if "text" not in df_real.columns or "label" not in df_real.columns:
            raise ValueError(
                f"{real_path} must contain 'text' and 'label' columns. "
                f"Found: {list(df_real.columns)}"
            )
    else:
        print(f"[INFO] No existing real dataset found at: {real_path}")
        print("[INFO] Starting from an empty real dataset.")
        df_real = pd.DataFrame(columns=["text", "label"])

    # ----- Load large_dataset (must exist) -----
    if not os.path.exists(large_path):
        raise FileNotFoundError(
            f"[ERROR] large_dataset.csv not found at: {large_path}\n"
            f"Please generate it first (e.g., via option 5 in main.py)."
        )

    print(f"[INFO] Loading large dataset from: {large_path}")
    df_large = pd.read_csv(large_path)

    if "text" not in df_large.columns or "label" not in df_large.columns:
        raise ValueError(
            f"{large_path} must contain 'text' and 'label' columns. "
            f"Found: {list(df_large.columns)}"
        )

    # ----- Basic cleaning -----
    df_real = df_real.dropna(subset=["text", "label"])
    df_large = df_large.dropna(subset=["text", "label"])

    df_real["text"] = df_real["text"].astype(str).str.strip()
    df_large["text"] = df_large["text"].astype(str).str.strip()

    df_real["label"] = df_real["label"].astype(int)
    df_large["label"] = df_large["label"].astype(int)

    print(f"[INFO] Real dataset size before merge: {len(df_real)}")
    print(f"[INFO] Large dataset size: {len(df_large)}")

    # ----- Concatenate -----
    df_merged = pd.concat([df_real, df_large], axis=0, ignore_index=True)

    # ----- Optional: drop duplicates by text+label -----
    if drop_duplicates:
        before = len(df_merged)
        df_merged = df_merged.drop_duplicates(subset=["text", "label"])
        after = len(df_merged)
        print(f"[INFO] Dropped {before - after} duplicate rows (by text+label).")

    # ----- Optional: shuffle rows -----
    if shuffle:
        df_merged = df_merged.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    # ----- Save back to real_dataset.csv -----
    os.makedirs(os.path.dirname(real_path), exist_ok=True)
    df_merged.to_csv(real_path, index=False, encoding="utf-8")

    print(f"[INFO] Saved merged dataset to: {real_path}")
    print(f"[INFO] Final merged size: {len(df_merged)}")
    print("[INFO] Label distribution:")
    print(df_merged["label"].value_counts())


if __name__ == "__main__":
    merge_large_into_real()
