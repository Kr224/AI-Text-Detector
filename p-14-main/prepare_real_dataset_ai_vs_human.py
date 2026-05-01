# prepare_real_dataset_ai_vs_human.py
"""
Convert the Kaggle 'AI Vs Human Text' dataset (AI_Human.csv)
into the unified format used by this project:

    data/processed/real_dataset.csv

Input file (AI_Human.csv):
    - text      : essay text
    - generated : 0 = human-written, 1 = AI-generated

Output file (real_dataset.csv):
    - text      : essay text (string)
    - label     : 0 = human, 1 = AI
"""

import os
import pandas as pd


def prepare_from_ai_vs_human(
    input_path: str = "data/processed/AI_vs_Human.csv",
    output_path: str = "data/processed/real_dataset.csv",
    max_samples: int = None,  # e.g. 50000 if you want a subset
):
    """Load AI_Human.csv -> rename columns -> optionally subsample -> save as real_dataset.csv."""

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    print(f"[INFO] Loading raw dataset from: {input_path}")
    # AI_Human.csv has ~487k rows and 2 columns: 'text' and 'generated'
    df = pd.read_csv(input_path)

    print("[INFO] Raw columns:", list(df.columns))
    # We expect columns: ['text', 'generated']
    if "text" not in df.columns or "generated" not in df.columns:
        raise KeyError("Expected columns 'text' and 'generated' not found in dataset.")

    # Basic cleaning: drop NaNs and strip whitespace
    df = df.dropna(subset=["text", "generated"])
    df["text"] = df["text"].astype(str).str.strip()

    # Filter out very short texts (optional, you can tune the threshold)
    df = df[df["text"].str.len() > 30]

    # Map 'generated' -> 'label'
    # According to the dataset description:
    #   generated = 0.0 or 0 -> human-written
    #   generated = 1.0 or 1 -> AI-generated
    df["label"] = df["generated"].astype(int)

    # Keep only the two columns we need
    df = df[["text", "label"]]

    # Optionally subsample to avoid training on the full 480k
    if max_samples is not None and len(df) > max_samples:
        print(f"[INFO] Subsampling from {len(df)} to {max_samples} samples (balanced by label)...")
        # Ensure roughly balanced classes
        dfs = []
        for label_value in [0, 1]:
            df_label = df[df["label"] == label_value]
            n = min(len(df_label), max_samples // 2)
            dfs.append(df_label.sample(n=n, random_state=42))
        df = pd.concat(dfs).sample(frac=1.0, random_state=42).reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")

    print(f"[INFO] Saved unified real dataset to: {output_path}")
    print(f"[INFO] Total samples: {len(df)}")
    print("[INFO] Label distribution:")
    print(df["label"].value_counts())


if __name__ == "__main__":
    # You can tune max_samples if needed. For full dataset, pass max_samples=None.
    prepare_from_ai_vs_human(
        max_samples=4000,  # e.g. use 50k samples for faster experiments
    )
