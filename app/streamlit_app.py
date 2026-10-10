"""House price estimator for King County, WA, using the saved Phase 2 pipeline.

Run:  streamlit run app/streamlit_app.py
"""

import sys
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import ZIP_PROFILE_FILE  # noqa: E402
from src.features import haversine_km  # noqa: E402
from src.predict import load_model, predict_frame  # noqa: E402

st.set_page_config(page_title="House Price Prediction Model", page_icon="🏠", layout="wide")


@st.cache_resource
def bundle():
    return load_model()


@st.cache_data
def zip_profile():
    return pd.read_csv(ZIP_PROFILE_FILE, dtype={"zipcode": str}).set_index("zipcode")


model, zips = bundle(), zip_profile()
hold = model["holdout"]

st.title("🏠 House Price Prediction Model")
st.caption(f"King County, WA (Seattle area). Estimates come from the saved **{model['name']}** pipeline, "
           f"trained on 2014–2015 sales: hold-out R² {hold['r2']:.3f}, median error {hold['median_ape']:.1%}.")

left, mid, right = st.columns([1, 1, 1.1], gap="large")
with left:
    st.subheader("Location and size")
    zipcode = st.selectbox("Zipcode", zips.index.tolist(), index=zips.index.get_loc("98103"),
                           format_func=lambda z: f"{z}  (median ${zips.loc[z, 'median_price']:,.0f})")
    sqft_living = st.number_input("Living area (sq ft)", 300, 13000, 1900, step=50)
    sqft_basement = st.number_input("of which basement (sq ft)", 0, 5000, 0, step=50)
    sqft_lot = st.number_input("Lot size (sq ft)", 500, 1_000_000, 5000, step=250)
    bedrooms = st.slider("Bedrooms", 0, 10, 3)
    bathrooms = st.slider("Bathrooms", 0.0, 8.0, 2.0, step=0.25)
    floors = st.select_slider("Floors", [1.0, 1.5, 2.0, 2.5, 3.0, 3.5], value=1.0)
with mid:
    st.subheader("Quality and history")
    grade = st.slider("Construction grade (King County scale, 1–13)", 1, 13, 7)
    condition = st.slider("Condition (1 poor – 5 very good)", 1, 5, 3)
    view = st.slider("View rating (0 none – 4 excellent)", 0, 4, 0)
    waterfront = st.checkbox("Waterfront")
    yr_built = st.number_input("Year built", 1900, 2015, 1975)
    renovated = st.checkbox("Renovated")
    yr_renovated = st.number_input("Year renovated", int(yr_built), 2015, max(int(yr_built), 2005),
                                   disabled=not renovated)
    with st.expander("Advanced: exact location and neighbourhood"):
        lat = st.number_input("Latitude", 47.15, 47.78, float(zips.loc[zipcode, "lat"]), format="%.4f")
        long = st.number_input("Longitude", -122.52, -121.31, float(zips.loc[zipcode, "long"]), format="%.4f")
        sqft_living15 = st.number_input("Avg living area of 15 nearest homes", 400, 6500,
                                        int(zips.loc[zipcode, "sqft_living15"]), step=50)
        sqft_lot15 = st.number_input("Avg lot size of 15 nearest homes", 600, 900_000,
                                     int(zips.loc[zipcode, "sqft_lot15"]), step=250)
        sale_date = st.date_input("Sale date", date(2015, 3, 1),
                                  min_value=date(2014, 5, 1), max_value=date(2015, 5, 31))

house = pd.DataFrame([{
    "date": pd.Timestamp(sale_date), "bedrooms": bedrooms, "bathrooms": bathrooms,
    "sqft_living": sqft_living, "sqft_lot": sqft_lot, "floors": floors, "waterfront": int(waterfront),
    "view": view, "condition": condition, "grade": grade,
    "sqft_above": max(sqft_living - sqft_basement, 1), "sqft_basement": min(sqft_basement, sqft_living - 1),
    "yr_built": yr_built, "yr_renovated": yr_renovated if renovated else 0, "zipcode": zipcode,
    "lat": lat, "long": long, "sqft_living15": sqft_living15, "sqft_lot15": sqft_lot15,
}])
price = float(predict_frame(house, model).iloc[0])
band = hold["median_ape"]
zip_median = zips.loc[zipcode, "median_price"]

with right:
    st.subheader("Estimate")
    st.metric("Estimated sale price", f"${price:,.0f}",
              delta=f"{price / zip_median - 1:+.0%} vs zipcode median", delta_color="off")
    st.markdown(f"**Typical range:** \\${price * (1 - band):,.0f} – \\${price * (1 + band):,.0f}  \n"
                f"<span style='color:gray;font-size:0.85em'>Half of hold-out houses were predicted within "
                f"±{band:.1%}.</span>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    c1.metric("Price per sq ft", f"${price / sqft_living:,.0f}")
    c2.metric("Zipcode median $/sq ft", f"${zips.loc[zipcode, 'median_price_per_sqft']:,.0f}")
    c1.metric("Distance to downtown", f"{float(haversine_km(lat, long)):.1f} km")
    c2.metric("Sales in zipcode", f"{int(zips.loc[zipcode, 'sales']):,}")
    st.info("Prices reflect the 2014–2015 market the model was trained on, not today's prices.", icon="ℹ️")
