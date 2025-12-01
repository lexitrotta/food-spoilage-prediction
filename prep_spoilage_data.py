# scripts/prep_spoilage_data.py

import pandas as pd
from pathlib import Path

RAW_PATH = Path("data/raw/spoilage_raw.csv")
PROC_PATH = Path("data/processed/spoilage_clean.csv")

def load_raw():
    return pd.read_csv(RAW_PATH)

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    # Example: binary event flag, duration in days
    if {"spoilage_day", "start_day"}.issubset(df.columns):
        df["duration"] = df["spoilage_day"] - df["start_day"]
        df["event"] = (df["is_spoiled"] == 1).astype(int)
    
    # Scale temperature or pH if helpful (leave as-is in template)
    return df

def main():
    df = load_raw()
    df_clean = engineer_features(df)
    PROC_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(PROC_PATH, index=False)
    print(f"Saved cleaned spoilage data to {PROC_PATH}")

if __name__ == "__main__":
    main()
