"""Regression metrics and the charts used in the README."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ACCENT = "#2a7f62"
MUTED = "#9bb8c9"


def adjusted_r2(r2: float, n: int, p: int) -> float:
    """Adjusted R²: penalises R² for the number of predictors p, given n rows."""
    return float(1 - (1 - r2) * (n - 1) / (n - p - 1))


def regression_metrics(y_true, y_pred, n_predictors: int) -> dict:
    """Dollar-scale metrics for one set of predictions."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    r2 = r2_score(y_true, y_pred)
    ape = np.abs(y_pred - y_true) / y_true
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2),
        "adj_r2": adjusted_r2(r2, len(y_true), n_predictors),
        "median_ape": float(np.median(ape)),
        "within_10pct": float(np.mean(ape <= 0.10)),
    }


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.3)


def plot_model_comparison(table: pd.DataFrame, path):
    """Hold-out RMSE and CV RMSE (mean ± std) per model."""
    t = table.sort_values("test_rmse", ascending=False)
    best = t["cv_rmse_mean"].idxmin()
    colors = [ACCENT if i == best else MUTED for i in t.index]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    axes[0].barh(t.index, t["test_rmse"] / 1000, color=colors)
    axes[0].set_xlabel("Hold-out RMSE, $ thousands")
    axes[1].barh(t.index, t["cv_rmse_mean"] / 1000, xerr=t["cv_rmse_std"] / 1000, color=colors)
    axes[1].set_xlabel("5-fold CV RMSE, $ thousands (mean ± std)")
    for ax, col, err in zip(axes, ["test_rmse", "cv_rmse_mean"], [None, "cv_rmse_std"]):
        _style(ax)
        for y, (v, e) in enumerate(zip(t[col] / 1000, (t[err] / 1000) if err else [0] * len(t))):
            ax.text(v + e, y, f"  {v:,.0f}", va="center", fontsize=9)
    fig.suptitle("Phase 2: model comparison (log-price target, engineered features)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_predicted_vs_actual(y_true, y_pred, title, path):
    fig, ax = plt.subplots(figsize=(6.2, 6))
    ax.scatter(y_true / 1e6, y_pred / 1e6, s=6, alpha=0.35, color=ACCENT, edgecolors="none")
    lim = [0.05, max(y_true.max(), y_pred.max()) / 1e6 * 1.05]
    ax.plot(lim, lim, "--", color="#555", lw=1, label="perfect prediction")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Actual sale price ($ millions, log scale)")
    ax.set_ylabel("Predicted price ($ millions, log scale)")
    ax.set_title(title)
    ax.legend(loc="upper left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_residuals(y_true, y_pred, title, path):
    """Residuals on the log scale (≈ percentage error) against the prediction."""
    resid = np.log(y_true) - np.log(y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [2, 1]})
    axes[0].scatter(y_pred / 1e6, resid, s=6, alpha=0.35, color=ACCENT, edgecolors="none")
    axes[0].axhline(0, color="#555", lw=1, ls="--")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Predicted price ($ millions, log scale)")
    axes[0].set_ylabel("log(actual) − log(predicted)")
    axes[0].set_title(title)
    axes[1].hist(resid, bins=60, color=MUTED, edgecolor="white")
    axes[1].axvline(0, color="#555", lw=1, ls="--")
    axes[1].set_xlabel("Log residual")
    axes[1].set_title("Residual distribution")
    for ax in axes:
        _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_barh(series: pd.Series, title, xlabel, path, signed=False):
    s = series.iloc[::-1]
    colors = [(ACCENT if v >= 0 else "#c0504d") if signed else ACCENT for v in s]
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(s) + 1.4))
    ax.barh(s.index, s.values, color=colors, xerr=None)
    if signed:
        ax.axvline(0, color="#555", lw=1)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame, path):
    cols = ["price", "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", "waterfront",
            "view", "condition", "grade", "sqft_above", "sqft_basement", "yr_built", "yr_renovated",
            "zipcode", "lat", "long", "sqft_living15", "sqft_lot15"]
    corr = df[cols].astype(float).corr()
    order = ["price"] + corr["price"].drop("price").sort_values(ascending=False).index.tolist()
    corr = corr.loc[order, order]
    cols = order
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    data = np.ma.array(corr.values, mask=mask)
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(data, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=60, ha="right")
    ax.set_yticks(range(len(cols)), cols)
    for i in range(len(cols)):
        for j in range(i + 1):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(v) > 0.55 else "#222")
    ax.set_title("Pearson correlation (features ordered by correlation with price)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
