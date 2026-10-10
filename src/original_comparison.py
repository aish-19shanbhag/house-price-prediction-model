"""Phase 1: the original 2020 course comparison, reproduced.

The 2020 notebook could not run top to bottom (undefined variables, a metrics table
that was never created, and one cell that logged another model's scores under the
Gradient Boosting name). This script rebuilds the comparison exactly as the notebook
specified each model, so the original conclusion can be checked:

* Simple Linear Regression on sqft_living
* Multiple Linear Regression, feature set 1 (6 features) and feature set 2 (10 features)
* Lasso Regression, alpha = 1, 100, 1000, on the 16 raw features
* Gradient Boosting, 400 trees, depth 5, learning rate 0.1, on sqft_living only

Same protocol as the notebook: raw price target, raw features (zipcode as a number),
train_test_split(train_size=0.8, random_state=3), and cross_val_score(cv=5) on the full
dataset. Run:  python -m src.original_comparison
"""

import json
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split

from src import config as C
from src.data_loader import load_raw
from src.evaluate import adjusted_r2

# The notebook ran Lasso with sklearn defaults on unscaled features (max_iter=1000) and
# silenced warnings. Coordinate descent does not fully converge on raw square footage and
# zipcode values; we keep the original settings for faithfulness and silence the same warning.
warnings.filterwarnings("ignore", message="Objective did not converge")


def original_models():
    """(name, setting, estimator, features) exactly as specified in the 2020 notebook."""
    rows = [
        ("Simple Linear Regression", "sqft_living only", LinearRegression(), C.ORIGINAL_SLR_FEATURES),
        ("Multiple Linear Regression-1", "6 selected features", LinearRegression(), C.ORIGINAL_MLR1_FEATURES),
        ("Multiple Linear Regression-2", "10 selected features", LinearRegression(), C.ORIGINAL_MLR2_FEATURES),
    ]
    for alpha in C.ORIGINAL_LASSO_ALPHAS:
        rows.append(("Lasso Regression", f"alpha={alpha}, 16 features",
                     Lasso(alpha=alpha), C.ORIGINAL_LASSO_FEATURES))
    gbr = GradientBoostingRegressor(n_estimators=400, max_depth=5, learning_rate=0.1,
                                    loss="squared_error", min_samples_split=2,
                                    random_state=C.ORIGINAL_SPLIT_SEED)
    rows.append(("Gradient Boosting", "400 trees, depth 5, sqft_living only", gbr, C.ORIGINAL_SLR_FEATURES))
    return rows


def run(df: pd.DataFrame | None = None) -> pd.DataFrame:
    df = load_raw() if df is None else df
    train, test = train_test_split(df, train_size=C.ORIGINAL_TRAIN_SIZE,
                                   random_state=C.ORIGINAL_SPLIT_SEED)
    records = []
    for name, setting, model, feats in original_models():
        model.fit(train[feats], train[C.TARGET])
        r2_tr = r2_score(train[C.TARGET], model.predict(train[feats]))
        pred = model.predict(test[feats])
        r2_te = r2_score(test[C.TARGET], pred)
        cv = cross_val_score(model, df[feats], df[C.TARGET], cv=C.ORIGINAL_CV_FOLDS, scoring="r2")
        records.append({
            "model": name,
            "setting": setting,
            "n_features": len(feats),
            "rmse_test": float(np.sqrt(mean_squared_error(test[C.TARGET], pred))),
            "r2_train": r2_tr,
            "adj_r2_train": adjusted_r2(r2_tr, len(train), len(feats)),
            "r2_test": r2_te,
            "adj_r2_test": adjusted_r2(r2_te, len(test), len(feats)),
            "cv5_r2_mean": float(cv.mean()),
            "cv5_r2_std": float(cv.std()),
        })
    # The notebook ranked its evaluation table by 5-fold cross-validation
    return pd.DataFrame(records).sort_values("cv5_r2_mean", ascending=False).reset_index(drop=True)


def plot(table: pd.DataFrame, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = table.sort_values("cv5_r2_mean")
    labels = [f"{m}\n({s})" for m, s in zip(t["model"], t["setting"])]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    best = t["cv5_r2_mean"].idxmax()
    colors = ["#2a7f62" if i == best else "#9bb8c9" for i in t.index]
    axes[0].barh(labels, t["cv5_r2_mean"], xerr=t["cv5_r2_std"], color=colors)
    axes[0].set_xlabel("5-fold CV R² (higher is better)")
    axes[1].barh(labels, t["rmse_test"] / 1000, color=colors)
    axes[1].set_xlabel("Test RMSE, $ thousands (lower is better)")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle("Phase 1: original 2020 course comparison, reproduced (raw price target)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    C.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    table = run()
    table.to_csv(C.METRICS_DIR / "phase1_original_comparison.csv", index=False)
    winner = table.iloc[0]
    summary = {
        "protocol": "raw price, raw features, train_test_split(train_size=0.8, random_state=3), "
                    "cross_val_score(cv=5) on all 21,613 rows",
        "winner_by_cv5_r2": f"{winner['model']} ({winner['setting']})",
        "winner_by_test_rmse": "{} ({})".format(*table.loc[table["rmse_test"].idxmin(), ["model", "setting"]]),
        "results": table.to_dict(orient="records"),
    }
    (C.METRICS_DIR / "phase1_original_comparison.json").write_text(json.dumps(summary, indent=2))
    plot(table, C.FIGURES_DIR / "phase1_model_comparison.png")

    pd.set_option("display.width", 200)
    fmt = table.copy()
    fmt["rmse_test"] = fmt["rmse_test"].map("${:,.0f}".format)
    print(fmt.round(3).to_string(index=False))
    print("\nWinner by 5-fold CV R²:", summary["winner_by_cv5_r2"])
    print("Winner by test RMSE:   ", summary["winner_by_test_rmse"])


if __name__ == "__main__":
    main()
