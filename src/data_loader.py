"""Load the King County house sales dataset."""

import pandas as pd

from src.config import DATA_FILE, RAW_INPUT_COLUMNS, TARGET


def load_raw(path=DATA_FILE) -> pd.DataFrame:
    """The CSV exactly as the 2020 notebook read it (zipcode stays numeric)."""
    return pd.read_csv(path)


def load_data(path=DATA_FILE) -> pd.DataFrame:
    """Typed copy for the 2026 pipeline: zipcode as a category label, date parsed."""
    df = pd.read_csv(path, dtype={"zipcode": str})
    df["date"] = pd.to_datetime(df["date"].str[:8], format="%Y%m%d")
    return df


def split_xy(df: pd.DataFrame):
    """Model inputs (raw columns only, no id) and the price target."""
    return df[RAW_INPUT_COLUMNS].copy(), df[TARGET].astype(float)
