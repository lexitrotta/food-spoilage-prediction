# scripts/train_regression_classification.py

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error

DATA_PATH = Path("data/processed/spoilage_clean.csv")

def load_data():
    return pd.read_csv(DATA_PATH)

def main():
    df = load_data()
    
    feature_cols = ["temp", "ph", "moisture"]  # adjust to your dataset
    X = df[feature_cols]
    
    # Classification: fresh vs spoiled (or spoil within X days)
    y_clf = df["is_spoiled"]  # or some label
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_clf, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Classification accuracy: {acc:.3f}")
    
    # Regression: predict spoilage day or remaining shelf-life
    if "duration" in df.columns:
        y_reg = df["duration"]
        X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.2, random_state=42)
        reg = RandomForestRegressor(random_state=42)
        reg.fit(X_train_r, y_train_r)
        pred_r = reg.predict(X_test_r)
        mae = mean_absolute_error(y_test_r, pred_r)
        rmse = mean_squared_error(y_test_r, pred_r, squared=False)
        print(f"Regression -> MAE={mae:.2f}, RMSE={rmse:.2f}")

if __name__ == "__main__":
    main()
