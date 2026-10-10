"""Feature engineering.

Every feature here is computed row by row from that house's own attributes, so the
step is stateless and cannot leak information across the train/test boundary.
Anything that learns from the data (scaling, zipcode encoding) lives inside the
sklearn Pipeline in src/models.py and is refitted within every CV fold.
"""

import numpy as np
import pandas as pd

from src.config import CATEGORICAL_FEATURES, DOWNTOWN_SEATTLE, NUMERIC_FEATURES

EARTH_RADIUS_KM = 6371.0


def haversine_km(lat, lon, lat0=DOWNTOWN_SEATTLE[0], lon0=DOWNTOWN_SEATTLE[1]):
    """Great-circle distance in kilometres from (lat, lon) to a reference point."""
    lat, lon = np.radians(np.asarray(lat, float)), np.radians(np.asarray(lon, float))
    lat0, lon0 = np.radians(lat0), np.radians(lon0)
    a = np.sin((lat - lat0) / 2) ** 2 + np.cos(lat) * np.cos(lat0) * np.sin((lon - lon0) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Raw King County columns in, model-ready columns out (NUMERIC + CATEGORICAL)."""
    out = pd.DataFrame(index=df.index)
    date = pd.to_datetime(df["date"])
    sale_year = date.dt.year

    for col in ["bedrooms", "bathrooms", "floors", "waterfront", "view", "condition",
                "grade", "lat", "long"]:
        out[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

    # Square footage is right-skewed; logs make the relationship with log-price closer to linear
    for col in ["sqft_living", "sqft_lot", "sqft_living15", "sqft_lot15"]:
        out[f"log_{col}"] = np.log1p(df[col].astype(float))

    out["basement_share"] = df["sqft_basement"].astype(float) / df["sqft_living"].clip(lower=1)
    out["dist_downtown_km"] = haversine_km(df["lat"], df["long"])

    yr_built = df["yr_built"].astype(int)
    yr_renov = df["yr_renovated"].astype(int)
    renovated = yr_renov > 0
    out["house_age"] = (sale_year - yr_built).clip(lower=0)
    out["is_renovated"] = renovated.astype(float)
    last_work = np.where(renovated, yr_renov, yr_built)
    out["years_since_renovation"] = np.clip(sale_year - last_work, 0, None)

    out["living_lot_ratio"] = df["sqft_living"].astype(float) / df["sqft_lot"].clip(lower=1)
    out["sale_year"] = sale_year.astype(float)
    out["sale_month"] = date.dt.month.astype(float)

    out["zipcode"] = df["zipcode"].astype(str)
    return out[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
