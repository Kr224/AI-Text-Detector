# prepare_real_dataset_from_ai_human.py
"""
Convert ai_human.csv from the GitHub dataset into the unified format:
    data/processed/real_dataset.csv

Input (ai_human.csv):
    - Text   : raw text
    - Labels : 0 = human, 1 = AI

Output (real_dataset.csv):
    - text   : raw text (string)
    - label  : 0 = human, 1 = AI
"""

import os
import pandas as pd


def prepare_from_ai_human(
    input_path: str = r"C:\Users\DELL\Downloads\ai_human.csv",
    output_path: str = "data/processed/real_dataset.csv",
):
    """Load ai_human.csv -> rename columns -> save as real_dataset.csv"""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    print(f"[INFO] Loading raw dataset from: {input_path}")
    df = pd.read_csv(input_path)

    print("[INFO] Raw columns:", list(df.columns))

    # These names are based on the GitHub README:
    # 'Text' column for the text, 'Labels' column for the label (0=human,1=AI)
    text_col = "Data"
    label_col = "Label"

    if text_col not in df.columns:
        raise KeyError(f"Data column '{text_col}' not found in dataset.")
    if label_col not in df.columns:
        raise KeyError(f"Label column '{label_col}' not found in dataset.")

    # Extract and normalize
    texts = df[text_col].astype(str)
    labels = df[label_col].astype(int)  # 0 = human, 1 = AI

    # Build unified DataFrame
    unified = pd.DataFrame({
        "text": texts,
        "label": labels,
    })

    # Basic cleaning: drop empty texts
    unified = unified.dropna(subset=["text", "label"])
    unified["text"] = unified["text"].astype(str).str.strip()
    unified["label"] = unified["label"].astype(int)

    # Filter out very short texts if needed
    unified = unified[unified["text"].str.len() > 30]

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    unified.to_csv(output_path, index=False, encoding="utf-8")

    print(f"[INFO] Saved unified real dataset to: {output_path}")
    print(f"[INFO] Total samples: {len(unified)}")
    print("[INFO] Label distribution:")
    print(unified["label"].value_counts())


if __name__ == "__main__":
    prepare_from_ai_human()
