"""Geospatial views of price: a static hexbin heat map and an interactive folium map."""

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

from src.config import DOWNTOWN_SEATTLE  # noqa: E402


def zipcode_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Per-zipcode medians, used for map labels and as default inputs in the demo app."""
    g = df.groupby("zipcode")
    prof = g.agg(sales=("price", "size"), median_price=("price", "median"),
                 lat=("lat", "median"), long=("long", "median"),
                 sqft_living15=("sqft_living15", "median"), sqft_lot15=("sqft_lot15", "median"))
    prof["median_price_per_sqft"] = g.apply(lambda d: (d["price"] / d["sqft_living"]).median(),
                                            include_groups=False)
    return prof.reset_index().sort_values("zipcode")


def plot_price_heatmap(df: pd.DataFrame, path):
    """Median sale price per hexagonal cell of latitude/longitude."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), gridspec_kw={"width_ratios": [1.15, 1]})
    hb = axes[0].hexbin(df["long"], df["lat"], C=df["price"], reduce_C_function=np.median,
                        gridsize=70, cmap="magma_r", norm=LogNorm(vmin=2e5, vmax=2e6), mincnt=2,
                        linewidths=0)
    cbar = fig.colorbar(hb, ax=axes[0], shrink=0.85)
    cbar.set_label("Median sale price ($, log scale)")
    axes[0].set_title("Median sale price by location")

    ppsf = df["price"] / df["sqft_living"]
    hb2 = axes[1].hexbin(df["long"], df["lat"], C=ppsf, reduce_C_function=np.median,
                         gridsize=70, cmap="viridis", mincnt=2, linewidths=0, vmin=100, vmax=500)
    cbar2 = fig.colorbar(hb2, ax=axes[1], shrink=0.85)
    cbar2.set_label("Median $ per sq ft of living area")
    axes[1].set_title("Price per square foot by location")

    for ax in axes:
        ax.plot(DOWNTOWN_SEATTLE[1], DOWNTOWN_SEATTLE[0], marker="*", ms=16, color="#00b4d8",
                mec="black", label="Downtown Seattle")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect(1 / np.cos(np.radians(47.5)))
        ax.legend(loc="lower left")
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle("King County house sales, May 2014 to May 2015 (21,613 sales)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def build_folium_map(df: pd.DataFrame, path) -> bool:
    """Interactive map: zipcode medians as coloured circles + a heat layer of the top 10% of sales.

    Returns False (and writes nothing) if folium is not installed.
    """
    try:
        import branca.colormap as cm
        import folium
        from folium.plugins import HeatMap
    except ImportError:
        return False

    prof = zipcode_profile(df)
    m = folium.Map(location=[47.50, -122.20], zoom_start=10, tiles="OpenStreetMap", control_scale=True)
    cmap = cm.LinearColormap(["#fcfdbf", "#fc8961", "#b73779", "#51127c", "#000004"],
                             vmin=200_000, vmax=1_200_000,
                             caption="Median sale price by zipcode ($)")
    zips = folium.FeatureGroup(name="Median price by zipcode").add_to(m)
    for _, r in prof.iterrows():
        folium.CircleMarker(
            location=[r["lat"], r["long"]], radius=5 + 6 * np.sqrt(r["sales"] / prof["sales"].max()),
            color="#333", weight=0.6, fill=True, fill_opacity=0.85,
            fill_color=cmap(min(r["median_price"], 1_200_000)),
            tooltip=(f"<b>{r['zipcode']}</b><br>median ${r['median_price']:,.0f}"
                     f"<br>${r['median_price_per_sqft']:,.0f} per sq ft<br>{r['sales']} sales"),
        ).add_to(zips)
    top = df[df["price"] >= df["price"].quantile(0.90)]
    HeatMap(top[["lat", "long"]].values.tolist(), radius=9, blur=12, min_opacity=0.25,
            name="Heat map: top 10% most expensive sales").add_to(m)
    peak = df.loc[df["price"].idxmax()]
    folium.Marker([peak["lat"], peak["long"]], tooltip=f"Highest sale: ${peak['price']:,.0f}",
                  icon=folium.Icon(color="green", icon="home")).add_to(m)
    cmap.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.save(str(path))
    return True
