"""Model zoo for the 2026 pipeline.

Each model is one sklearn Pipeline:

    raw columns -> engineer_features (stateless) -> ColumnTransformer -> regressor
    wrapped in TransformedTargetRegressor(log1p / expm1)

so the regressor learns log(price) while every prediction comes back in dollars, and
everything that is fitted (scaler, zipcode encoder, alpha search) is refitted inside
each cross-validation fold.
"""

import numpy as np
import sklearn
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler, TargetEncoder
from xgboost import XGBRegressor

from src.config import CATEGORICAL_FEATURES, CV_FOLDS, NUMERIC_FEATURES, RANDOM_STATE
from src.features import engineer_features

def _engineer():
    return FunctionTransformer(engineer_features, validate=False)


def linear_preprocessor():
    """Standardised numerics + one-hot zipcode (70 levels; unseen zipcodes -> all zeros)."""
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("zip", OneHotEncoder(handle_unknown="ignore", drop=None, sparse_output=False), CATEGORICAL_FEATURES),
    ], verbose_feature_names_out=False)


def _target_encoder():
    """TargetEncoder with seeded internal cross-fitting (API changed in scikit-learn 1.9)."""
    if tuple(int(v) for v in sklearn.__version__.split(".")[:2]) >= (1, 9):
        folds = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        return TargetEncoder(target_type="continuous", cv=folds)
    return TargetEncoder(target_type="continuous", cv=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def tree_preprocessor():
    """Numerics as-is + cross-fitted target encoding of zipcode (one column)."""
    return ColumnTransformer([
        ("num", "passthrough", NUMERIC_FEATURES),
        ("zip", _target_encoder(), CATEGORICAL_FEATURES),
    ], verbose_feature_names_out=False)


def simple_preprocessor():
    """Simple linear regression: one input, log living area."""
    return ColumnTransformer([("num", "passthrough", ["log_sqft_living"])], verbose_feature_names_out=False)


def _wrap(pre, reg):
    pipe = Pipeline([("features", _engineer()), ("prep", pre), ("model", reg)])
    return TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)


def build_models() -> dict:
    """Name -> unfitted estimator. Hyperparameters are fixed, sensible defaults, not tuned on the test set."""
    inner_cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    return {
        "Simple Linear Regression": _wrap(simple_preprocessor(), LinearRegression()),
        "Multiple Linear Regression": _wrap(linear_preprocessor(), LinearRegression()),
        "Ridge (alpha by CV)": _wrap(linear_preprocessor(), RidgeCV(alphas=np.logspace(-3, 3, 25))),
        "Lasso (alpha by CV)": _wrap(linear_preprocessor(),
                                     LassoCV(alphas=np.logspace(-7, -1, 49), cv=inner_cv, max_iter=20_000,
                                             random_state=RANDOM_STATE)),
        "Gradient Boosting": _wrap(tree_preprocessor(), GradientBoostingRegressor(
            # same tree budget as the 2020 course model (400 trees, depth 5, lr 0.1)
            n_estimators=400, learning_rate=0.1, max_depth=5, subsample=0.8,
            min_samples_leaf=5, random_state=RANDOM_STATE)),
        "HistGradientBoosting": _wrap(tree_preprocessor(), HistGradientBoostingRegressor(
            max_iter=800, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=20,
            l2_regularization=1.0, random_state=RANDOM_STATE)),
        "XGBoost": _wrap(tree_preprocessor(), XGBRegressor(
            n_estimators=800, learning_rate=0.05, max_depth=6, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=3, reg_lambda=1.0,
            n_jobs=4, random_state=RANDOM_STATE)),
    }


def n_model_inputs(fitted) -> int:
    """Number of columns the regressor actually sees (used for adjusted R²)."""
    pipe = fitted.regressor_
    return len(pipe.named_steps["prep"].get_feature_names_out())


def inner_model(fitted):
    return fitted.regressor_.named_steps["model"]


def model_input_names(fitted):
    return list(fitted.regressor_.named_steps["prep"].get_feature_names_out())
