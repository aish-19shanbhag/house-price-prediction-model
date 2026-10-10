"""Phase 2: the 2026 pipeline. Train, cross-validate, evaluate on a hold-out set, save.

Protocol
* 80/20 hold-out split (random_state=42). The 20% is touched once, at the end.
* 5-fold shuffled CV on the 80% for every model; the deployed model is the one with the
  lowest mean CV RMSE (chosen without looking at the hold-out set).
* Target is log(price); every metric is computed after converting back to dollars.

Run:  python -m src.train
"""

import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold, cross_validate, train_test_split

from src import config as C
from src.data_loader import load_data, split_xy
from src.evaluate import (plot_barh, plot_correlation_heatmap, plot_model_comparison,
                          plot_predicted_vs_actual, plot_residuals, regression_metrics)
from src.geo import build_folium_map, plot_price_heatmap, zipcode_profile
from src.models import build_models, inner_model, model_input_names, n_model_inputs

SCORING = {"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error", "r2": "r2"}


def log(msg, lines):
    print(msg, flush=True)
    lines.append(msg)


def main():
    for d in (C.FIGURES_DIR, C.METRICS_DIR, C.MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    lines = []
    df = load_data()
    X, y = split_xy(df)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=C.TEST_SIZE, random_state=C.RANDOM_STATE)
    log(f"rows: train {len(X_tr):,}  hold-out {len(X_te):,}", lines)

    cv = KFold(n_splits=C.CV_FOLDS, shuffle=True, random_state=C.RANDOM_STATE)
    results, fitted = {}, {}
    for name, model in build_models().items():
        t0 = time.time()
        scores = cross_validate(model, X_tr, y_tr, cv=cv, scoring=SCORING, n_jobs=1)
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        p = n_model_inputs(model)
        hold = regression_metrics(y_te, pred, p)
        results[name] = {
            "n_model_inputs": p,
            "holdout": hold,
            "cv": {
                "rmse_mean": float(-scores["test_rmse"].mean()), "rmse_std": float(scores["test_rmse"].std()),
                "mae_mean": float(-scores["test_mae"].mean()), "mae_std": float(scores["test_mae"].std()),
                "r2_mean": float(scores["test_r2"].mean()), "r2_std": float(scores["test_r2"].std()),
            },
            "seconds": round(time.time() - t0, 1),
        }
        fitted[name] = (model, pred)
        log(f"{name:28s} hold-out RMSE ${hold['rmse']:>9,.0f}  MAE ${hold['mae']:>8,.0f}  "
            f"R2 {hold['r2']:.4f}  adjR2 {hold['adj_r2']:.4f} | CV RMSE ${results[name]['cv']['rmse_mean']:,.0f}"
            f" ± {results[name]['cv']['rmse_std']:,.0f}  CV R2 {results[name]['cv']['r2_mean']:.4f}"
            f" ± {results[name]['cv']['r2_std']:.4f}  ({results[name]['seconds']}s)", lines)

    table = pd.DataFrame({
        n: {"test_rmse": r["holdout"]["rmse"], "test_mae": r["holdout"]["mae"], "test_r2": r["holdout"]["r2"],
            "test_adj_r2": r["holdout"]["adj_r2"], "median_ape": r["holdout"]["median_ape"],
            "within_10pct": r["holdout"]["within_10pct"],
            "cv_rmse_mean": r["cv"]["rmse_mean"], "cv_rmse_std": r["cv"]["rmse_std"],
            "cv_r2_mean": r["cv"]["r2_mean"], "cv_r2_std": r["cv"]["r2_std"], "n_model_inputs": r["n_model_inputs"]}
        for n, r in results.items()}).T
    table.index.name = "model"
    table.to_csv(C.METRICS_DIR / "phase2_results.csv")

    best_name = table["cv_rmse_mean"].astype(float).idxmin()
    linear = [n for n in table.index if "Linear" in n or "Lasso" in n or "Ridge" in n]
    best_linear = table.loc[linear, "cv_rmse_mean"].astype(float).idxmin()
    best_model, best_pred = fitted[best_name]
    lasso_name = "Lasso (alpha by CV)"
    lasso_model = fitted[lasso_name][0]
    lasso_reg = inner_model(lasso_model)

    def reduction(a, b):
        return float(1 - table.loc[a, "test_rmse"] / table.loc[b, "test_rmse"])

    summary = {
        "best_model_by_cv_rmse": best_name,
        "best_linear_model_by_cv_rmse": best_linear,
        "rmse_reduction_best_vs_lasso": reduction(best_name, lasso_name),
        "rmse_reduction_best_vs_simple_linear": reduction(best_name, "Simple Linear Regression"),
        "rmse_reduction_lasso_vs_simple_linear": reduction(lasso_name, "Simple Linear Regression"),
        "lasso_alpha": float(lasso_reg.alpha_),
        "lasso_nonzero_coefs": int(np.sum(lasso_reg.coef_ != 0)),
        "lasso_total_coefs": int(lasso_reg.coef_.size),
    }
    log(f"\nbest model by CV RMSE: {best_name}; best linear model: {best_linear}", lines)
    log(f"hold-out RMSE reduction, {best_name} vs Lasso: {summary['rmse_reduction_best_vs_lasso']:.1%}", lines)
    log(f"Lasso alpha {summary['lasso_alpha']:.2e}, non-zero coefficients "
        f"{summary['lasso_nonzero_coefs']}/{summary['lasso_total_coefs']}", lines)

    # Lasso coefficients (standardised inputs, log-price units)
    coefs = pd.Series(lasso_reg.coef_, index=model_input_names(lasso_model), name="coef")
    coefs.sort_values(key=np.abs, ascending=False).to_csv(C.METRICS_DIR / "lasso_coefficients.csv")
    numeric = coefs[[c for c in coefs.index if not c.startswith("zipcode_")]]
    plot_barh(numeric.sort_values(key=np.abs, ascending=False),
              "Lasso coefficients (non-zipcode features)",
              "Change in log(price) per 1 std increase (≈ % change)",
              C.FIGURES_DIR / "lasso_coefficients.png", signed=True)

    # Permutation importance of the deployed model, on the raw input columns
    perm = permutation_importance(best_model, X_te, y_te, scoring="neg_root_mean_squared_error",
                                  n_repeats=5, random_state=C.RANDOM_STATE, n_jobs=1)
    imp = pd.DataFrame({"rmse_increase_mean": perm.importances_mean, "rmse_increase_std": perm.importances_std},
                       index=X_te.columns).sort_values("rmse_increase_mean", ascending=False)
    imp.to_csv(C.METRICS_DIR / "permutation_importance.csv")
    plot_barh(imp["rmse_increase_mean"].head(15),
              f"Permutation importance, {best_name} (hold-out set)",
              "Increase in RMSE ($) when the column is shuffled",
              C.FIGURES_DIR / "permutation_importance.png")

    # Figures
    plot_model_comparison(table.astype({c: float for c in table.columns}), C.FIGURES_DIR / "phase2_model_comparison.png")
    plot_predicted_vs_actual(y_te.values, best_pred, f"{best_name}: predicted vs actual (hold-out)",
                             C.FIGURES_DIR / "predicted_vs_actual.png")
    plot_residuals(y_te.values, best_pred, f"{best_name}: residuals (hold-out)", C.FIGURES_DIR / "residuals.png")
    plot_correlation_heatmap(df, C.FIGURES_DIR / "correlation_heatmap.png")
    plot_price_heatmap(df, C.FIGURES_DIR / "geo_price_heatmap.png")
    folium_ok = build_folium_map(df, C.FIGURES_DIR / "price_map.html")
    log(f"folium map written: {folium_ok}", lines)

    # Demo assets: per-zipcode defaults and a handful of hold-out houses
    zipcode_profile(df).round(4).to_csv(C.ZIP_PROFILE_FILE, index=False)
    raw = pd.read_csv(C.DATA_FILE, dtype={"zipcode": str})
    raw.loc[X_te.index[:12]].to_csv(C.DATA_DIR / "sample_houses.csv", index=False)

    metrics = {"protocol": {"split": "80/20 hold-out, random_state=42", "cv": "5-fold shuffled KFold on the 80%",
                            "target": "log1p(price); metrics in dollars after expm1",
                            "selection": "deployed model = lowest mean CV RMSE",
                            "train_rows": len(X_tr), "holdout_rows": len(X_te)},
               "summary": summary, "results": results}
    (C.METRICS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    joblib.dump({"model": best_model, "name": best_name,
                 "holdout": results[best_name]["holdout"], "cv": results[best_name]["cv"]},
                C.MODEL_FILE, compress=3)
    log(f"saved {C.MODEL_FILE.relative_to(C.ROOT)}", lines)
    (C.METRICS_DIR / "train_log.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
