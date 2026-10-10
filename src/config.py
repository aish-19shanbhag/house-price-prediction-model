"""Project-wide constants: paths, dataset schema, and experiment settings."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "reports" / "figures"
METRICS_DIR = ROOT / "reports" / "metrics"

DATA_FILE = DATA_DIR / "kc_house_data.csv"
MODEL_FILE = MODELS_DIR / "house_price_model.joblib"
ZIP_PROFILE_FILE = DATA_DIR / "zipcode_profile.csv"

TARGET = "price"

# ---------------------------------------------------------------------------
# Phase 1: the original 2020 course comparison (kept exactly as specified)
# ---------------------------------------------------------------------------
ORIGINAL_SPLIT_SEED = 3          # train_test_split(..., train_size=0.8, random_state=3)
ORIGINAL_TRAIN_SIZE = 0.8
ORIGINAL_CV_FOLDS = 5            # cross_val_score(model, X_all, y_all, cv=5)

ORIGINAL_SLR_FEATURES = ["sqft_living"]
ORIGINAL_MLR1_FEATURES = ["bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", "zipcode"]
ORIGINAL_MLR2_FEATURES = ["bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", "view",
                          "grade", "zipcode", "sqft_living15", "sqft_lot15"]
ORIGINAL_LASSO_FEATURES = ["bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", "waterfront",
                           "view", "condition", "grade", "sqft_above", "sqft_basement",
                           "zipcode", "lat", "long", "sqft_living15", "sqft_lot15"]
ORIGINAL_LASSO_ALPHAS = [1, 100, 1000]

# ---------------------------------------------------------------------------
# Phase 2: the 2026 pipeline
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# Downtown Seattle (Pioneer Square / city centre), used for a distance feature
DOWNTOWN_SEATTLE = (47.6062, -122.3321)

# Columns a user (or the app) must supply to price one house
RAW_INPUT_COLUMNS = ["date", "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
                     "waterfront", "view", "condition", "grade", "sqft_above", "sqft_basement",
                     "yr_built", "yr_renovated", "zipcode", "lat", "long",
                     "sqft_living15", "sqft_lot15"]

# Numeric model inputs after feature engineering (zipcode is handled separately)
NUMERIC_FEATURES = ["bedrooms", "bathrooms", "floors", "waterfront", "view", "condition", "grade",
                    "log_sqft_living", "log_sqft_lot", "log_sqft_living15", "log_sqft_lot15",
                    "basement_share", "lat", "long", "dist_downtown_km",
                    "house_age", "is_renovated", "years_since_renovation",
                    "living_lot_ratio", "sale_year", "sale_month"]
CATEGORICAL_FEATURES = ["zipcode"]
