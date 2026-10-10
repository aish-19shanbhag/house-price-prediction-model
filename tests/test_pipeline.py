"""Fast checks on a small sample: run with  pytest -q"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from src.config import CATEGORICAL_FEATURES, MODEL_FILE, NUMERIC_FEATURES, RAW_INPUT_COLUMNS
from src.data_loader import load_data, load_raw, split_xy
from src.evaluate import adjusted_r2, regression_metrics
from src.features import engineer_features, haversine_km
from src.models import build_models, n_model_inputs
from src.original_comparison import original_models

DF = load_data()
SAMPLE = DF.sample(1500, random_state=0)


def test_dataset_shape_and_no_missing_values():
    assert DF.shape == (21613, 21)
    assert DF[RAW_INPUT_COLUMNS + ["price"]].isna().sum().sum() == 0
    assert DF["zipcode"].nunique() == 70


def test_engineered_features():
    X, _ = split_xy(SAMPLE)
    F = engineer_features(X)
    assert list(F.columns) == NUMERIC_FEATURES + CATEGORICAL_FEATURES
    assert F[NUMERIC_FEATURES].isna().sum().sum() == 0
    assert (F["house_age"] >= 0).all()
    assert set(F["is_renovated"].unique()) <= {0.0, 1.0}
    assert (F["years_since_renovation"] <= F["house_age"]).all()
    assert F["dist_downtown_km"].between(0, 80).all()


def test_haversine_known_distance():
    # Downtown Seattle to Bellevue downtown is roughly 10 km
    assert haversine_km(47.6101, -122.2015) == pytest.approx(9.8, abs=1.0)
    assert haversine_km(47.6062, -122.3321) == pytest.approx(0.0, abs=1e-6)


def test_adjusted_r2():
    assert adjusted_r2(0.8, 101, 0) == pytest.approx(0.8)
    assert adjusted_r2(0.8, 101, 10) == pytest.approx(1 - 0.2 * 100 / 90)
    m = regression_metrics([100.0, 200.0, 300.0, 400.0], [110.0, 190.0, 300.0, 400.0], 1)
    assert m["mae"] == pytest.approx(5.0)
    assert m["within_10pct"] == 1.0


def test_original_comparison_matches_notebook_spec():
    specs = {(n, s): (m, f) for n, s, m, f in original_models()}
    gbr, gbr_feats = specs[("Gradient Boosting", "400 trees, depth 5, sqft_living only")]
    assert gbr.n_estimators == 400 and gbr.max_depth == 5 and gbr_feats == ["sqft_living"]
    lasso_alphas = sorted(m.alpha for (n, _), (m, f) in specs.items() if n == "Lasso Regression")
    assert lasso_alphas == [1, 100, 1000]
    assert all(len(f) == 16 for (n, _), (m, f) in specs.items() if n == "Lasso Regression")
    assert isinstance(specs[("Multiple Linear Regression-1", "6 selected features")][0], LinearRegression)
    assert load_raw()["zipcode"].dtype.kind == "i"  # the notebook used zipcode as a raw number


@pytest.mark.parametrize("name", ["Lasso (alpha by CV)", "HistGradientBoosting"])
def test_pipeline_trains_predicts_dollars_and_handles_unseen_zipcode(name):
    X, y = split_xy(SAMPLE)
    model = build_models()[name].fit(X, y)
    pred = model.predict(X)
    assert pred.min() > 10_000  # dollars, not log-dollars
    assert np.corrcoef(pred, y)[0, 1] > 0.8
    new = X.head(3).copy()
    new["zipcode"] = "99999"
    assert np.isfinite(model.predict(new)).all()
    assert n_model_inputs(model) >= len(NUMERIC_FEATURES) + 1


def test_saved_model_scores_sample_houses():
    if not MODEL_FILE.exists():
        pytest.skip("run `python -m src.train` first")
    from src.predict import load_model, predict_frame

    bundle = load_model()
    sample = pd.read_csv(MODEL_FILE.parents[1] / "data" / "sample_houses.csv", dtype={"zipcode": str})
    pred = predict_frame(sample, bundle)
    ape = (pred - sample["price"]).abs() / sample["price"]
    assert len(pred) == len(sample)
    assert ape.median() < 0.25
