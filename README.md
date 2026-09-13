# House Price Prediction Model

A multi-model regression pipeline for predicting house prices on the King County (Seattle area) housing dataset, comparing Linear, Multiple Linear, Lasso, and Gradient Boosting regression approaches.

## Overview

This project analyzes over 20,000 home sales records to predict sale price from property features. Beyond fitting a single model, the project's focus was comparing regression approaches head-to-head and using geospatial and statistical analysis to understand what actually drives price in this market.

## My Contribution

This was a 4-person academic project. My specific contributions were:
- Building the **Multiple Linear Regression** model
- Contributing to overall **model evaluation**
- Writing the **project conclusion**

## Key Features

- **15+ engineered features**: square footage, bedrooms, bathrooms, lot size, floors, waterfront, view, condition, grade, and location (zip code, lat/long)
- **Geospatial visualization**: heat maps built with Folium to visualize regional pricing patterns across King County
- **Multi-model comparison**: Linear Regression, Multiple Linear Regression, Lasso Regression, and Gradient Boosting Regressor evaluated side by side
- **Statistical rigor**: RMSE, R², and adjusted R² used for model evaluation, with k-fold cross-validation

## Results

Lasso Regression produced the lowest RMSE and highest R² among the models tested, making it the most reliable model for this dataset.

## Tech Stack

- Python (Pandas, NumPy, Scikit-learn)
- Matplotlib / Seaborn / Folium for visualization
- Jupyter Notebook

## Files

- `house_price_prediction.ipynb` — full analysis and modeling pipeline
- `kc_house_data.csv` — King County housing dataset

## Running the Project

```bash
pip install pandas numpy scikit-learn matplotlib seaborn folium jupyter
jupyter notebook house_price_prediction.ipynb
```

## Author

Aishwarya Ramanath Shanbhag
