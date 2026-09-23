# California Housing Value Prediction

An end-to-end Machine Learning case study for predicting California median house values using census-tract housing data.

The project covers the complete Data Science workflow — from data quality auditing and exploratory analysis to feature engineering, model comparison, hyperparameter tuning, explainability, robustness testing, model serialization, batch inference, and simulated production monitoring.

---

## 🎯 Business Problem

Real estate valuation is an important process for financial institutions, real estate marketplaces, urban planners, lenders, and government organizations.

The objective of this project is to build a predictive model capable of estimating the **median house value** of California census tracts using demographic, housing, economic, and geographic information.

### Business Objectives

- Reduce prediction error in house valuation.
- Identify the major factors influencing predicted house values.
- Build an interpretable and reproducible ML pipeline.
- Validate model robustness across different data splits.
- Prepare the model for deployment and batch inference.
- Establish a framework for monitoring data and prediction drift.

---

## 📊 Dataset

**Dataset:** California Housing Prices

**Source:** Kaggle

**Rows:** 20,640

**Original Features:** 9 predictors + 1 target

### Original Columns

| Feature | Description |
|---|---|
| `longitude` | Geographic longitude |
| `latitude` | Geographic latitude |
| `housing_median_age` | Median age of houses |
| `total_rooms` | Total number of rooms |
| `total_bedrooms` | Total number of bedrooms |
| `population` | Population in the block group |
| `households` | Number of households |
| `median_income` | Median income |
| `ocean_proximity` | Proximity to the ocean |
| `median_house_value` | Target variable |

### Data Quality

- 20,640 observations
- 10 original columns
- 207 missing values in `total_bedrooms`
- No duplicate rows identified
- `ocean_proximity` contains 5 categories
- Potential outliers were analyzed using the IQR method
- The target contains a concentration at `$500,001`, indicating an upper-value cap in the recorded data

---

## 🔬 Data Science Workflow

```text
Raw Dataset
     │
     ▼
Data Ingestion
     │
     ▼
Data Quality Audit
     │
     ├── Missing Values
     ├── Duplicate Detection
     ├── Data Types
     └── Outlier Analysis
     │
     ▼
Exploratory Data Analysis
     │
     ├── Distributions
     ├── Correlation
     ├── Spatial Analysis
     └── Categorical Analysis
     │
     ▼
Feature Engineering
     │
     ├── rooms_per_household
     ├── bedrooms_per_room
     ├── population_per_household
     ├── Latitude Bins
     └── Longitude Bins
     │
     ▼
Preprocessing
     │
     ├── Missing Value Imputation
     ├── Standard Scaling
     └── One-Hot Encoding
     │
     ▼
Model Development
     │
     ├── Linear Regression
     ├── Ridge
     ├── Lasso
     ├── SVR
     ├── KNN
     ├── Decision Tree
     ├── Random Forest
     └── XGBoost
     │
     ▼
Hyperparameter Tuning
     │
     ▼
Model Evaluation
     │
     ├── RMSE
     ├── MAE
     ├── R²
     └── MAPE
     │
     ▼
Explainability
     │
     ├── Feature Importance
     └── SHAP
     │
     ▼
Robustness Analysis
     │
     ├── Cross-Validation
     ├── Learning Curves
     └── Multiple Random Splits
     │
     ▼
Deployment Pipeline
     │
     ├── Serialization
     ├── Batch Inference
     └── Monitoring / Drift Detection
