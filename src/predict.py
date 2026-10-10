"""Price new houses with the saved pipeline.

Usage:  python -m src.predict path/to/houses.csv
The CSV needs the raw King County columns listed in src/config.RAW_INPUT_COLUMNS.
"""

import sys

import joblib
import pandas as pd

from src.config import MODEL_FILE, RAW_INPUT_COLUMNS


def load_model(path=MODEL_FILE):
    """Returns the bundle saved by src.train: {'model', 'name', 'metrics', ...}."""
    return joblib.load(path)


def predict_frame(df: pd.DataFrame, bundle=None) -> pd.Series:
    bundle = bundle or load_model()
    missing = [c for c in RAW_INPUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    X = df[RAW_INPUT_COLUMNS].copy()
    X["zipcode"] = X["zipcode"].astype(str)
    return pd.Series(bundle["model"].predict(X), index=df.index, name="predicted_price")


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    df = pd.read_csv(sys.argv[1], dtype={"zipcode": str})
    out = df.assign(predicted_price=predict_frame(df).round(-3))
    cols = ["zipcode", "sqft_living", "bedrooms", "grade"] + (["price"] if "price" in out else []) + ["predicted_price"]
    print(out[cols].to_string(index=False))


if __name__ == "__main__":
    main()
