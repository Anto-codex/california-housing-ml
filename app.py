from pathlib import Path

app_code = r'''# =============================================================================
# California Housing Value Predictor
# End-to-End Data Science Case Study — Streamlit Dashboard
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
import joblib

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="California Housing Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# CUSTOM CSS
# =============================================================================

st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf1 100%);
    }

    .hero {
        background: linear-gradient(120deg, #1e3c72 0%, #2a5298 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px rgba(30, 60, 114, 0.25);
    }

    .hero h1 {
        font-size: 2.4rem;
        font-weight: 700;
        margin: 0 0 0.4rem 0;
        color: white;
    }

    .hero p {
        font-size: 1.05rem;
        opacity: 0.95;
        margin: 0;
    }

    .metric-card {
        background: white;
        padding: 1.3rem 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border-left: 5px solid #2a5298;
        height: 100%;
    }

    .metric-label {
        font-size: 0.8rem;
        color: #6b7280;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.6px;
        margin-bottom: 0.3rem;
    }

    .metric-value {
        font-size: 1.9rem;
        font-weight: 700;
        color: #1e3c72;
        line-height: 1.1;
    }

    .section-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1e3c72;
        border-bottom: 3px solid #2a5298;
        padding-bottom: 0.4rem;
        margin: 1.5rem 0 1rem 0;
        display: inline-block;
    }

    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 14px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.35);
        margin: 1rem 0;
    }

    .prediction-value {
        font-size: 3rem;
        font-weight: 800;
        margin: 0.4rem 0;
    }

    .prediction-label {
        font-size: 0.95rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1.2px;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%);
    }

    [data-testid="stSidebar"] * {
        color: white !important;
    }

    .small-note {
        color: #6b7280;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# CONSTANTS
# =============================================================================

DATA_PATH = Path("data/housing.csv")
MODEL_PATH = Path("models/california_housing_xgb.joblib")

# These are the exact training-time boundaries used for spatial discretization.
LATITUDE_BINS = [
    32.53059, 33.481, 34.422, 35.363, 36.304,
    37.245, 38.186, 39.127, 40.068, 41.009, 41.95
]

LONGITUDE_BINS = [
    -124.36004, -123.346, -122.342, -121.338, -120.334,
    -119.33, -118.326, -117.322, -116.318, -115.314, -114.31
]

FINAL_METRICS = {
    "RMSE": 44409.8058,
    "MAE": 28872.7078,
    "R2": 0.849495,
    "MAPE": 16.2042,
}

# =============================================================================
# LOADERS
# =============================================================================

@st.cache_data(show_spinner=False)
def load_data():
    if not DATA_PATH.exists():
        return None
    return pd.read_csv(DATA_PATH)


@st.cache_resource(show_spinner=False)
def load_pipeline():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def engineer_features(input_df):
    """Reproduce the feature engineering used during model development."""
    df = input_df.copy()

    # Avoid division errors. The trained pipeline handles resulting NaNs.
    df["rooms_per_household"] = np.where(
        df["households"] > 0,
        df["total_rooms"] / df["households"],
        np.nan,
    )

    df["bedrooms_per_room"] = np.where(
        df["total_rooms"] > 0,
        df["total_bedrooms"] / df["total_rooms"],
        np.nan,
    )

    df["population_per_household"] = np.where(
        df["households"] > 0,
        df["population"] / df["households"],
        np.nan,
    )

    df["latitude_bin"] = pd.cut(
        df["latitude"],
        bins=LATITUDE_BINS,
        labels=False,
        include_lowest=True,
    )

    df["longitude_bin"] = pd.cut(
        df["longitude"],
        bins=LONGITUDE_BINS,
        labels=False,
        include_lowest=True,
    )

    return df


# =============================================================================
# HERO
# =============================================================================

st.markdown(
    """
    <div class="hero">
        <h1>🏠 California Housing Value Predictor</h1>
        <p>End-to-End Data Science Case Study — Predicting Median House Values</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# SIDEBAR NAVIGATION
# =============================================================================

with st.sidebar:
    st.markdown("## 🧭 Navigation")

    page = st.radio(
        "Go to",
        [
            "🏠 Overview",
            "📊 EDA",
            "🤖 Model Comparison",
            "🎯 Predict",
            "💡 Insights",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### 📌 Quick Facts")
    st.markdown(
        """
        - **Dataset:** 20,640 rows
        - **Target:** `median_house_value`
        - **Final Model:** Tuned XGBoost
        - **R²:** 0.8495
        - **RMSE:** $44.4K
        """
    )
    st.markdown("---")
    st.caption("Built with Streamlit • XGBoost • Plotly")


# =============================================================================
# PAGE 1 — OVERVIEW
# =============================================================================

def page_overview(df):
    st.markdown('<div class="section-title">📊 Model Performance</div>',
                unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    metrics = [
        ("Final Model", "Tuned XGBoost"),
        ("Test R²", "0.8495"),
        ("RMSE", "$44.4K"),
        ("MAPE", "16.20%"),
    ]

    for col, (label, value) in zip([c1, c2, c3, c4], metrics):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns([2, 1])

    with left:
        st.markdown('<div class="section-title">📌 Project Overview</div>',
                    unsafe_allow_html=True)
        st.markdown(
            """
            This application presents an end-to-end Machine Learning case study
            for predicting California median house values.

            **Workflow covered:**
            - Data quality auditing
            - Exploratory data analysis
            - Feature engineering
            - Categorical encoding
            - Spatial feature engineering
            - Model comparison and tuning
            - Explainability
            - Robustness analysis
            - Model serialization and batch inference
            - Simulated drift monitoring
            """
        )

    with right:
        st.markdown('<div class="section-title">🎯 Business Targets</div>',
                    unsafe_allow_html=True)
        st.markdown(
            """
            | Metric | Target |
            |---|---:|
            | RMSE | < $60K |
            | R² | > 0.75 |
            | MAPE | < 15% |
            """
        )

        st.warning(
            "MAPE is 16.20%, so the MAPE target remains slightly above the "
            "case-study threshold."
        )

    if df is not None:
        st.markdown('<div class="section-title">👀 Data Preview</div>',
                    unsafe_allow_html=True)
        st.dataframe(df.head(10), use_container_width=True)


# =============================================================================
# PAGE 2 — EDA
# =============================================================================

def page_eda(df):
    if df is None:
        st.error("Dataset not found at `data/housing.csv`.")
        return

    st.markdown('<div class="section-title">📊 Exploratory Data Analysis</div>',
                unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🔍 EDA Filters")

        prox = st.multiselect(
            "Ocean Proximity",
            options=df["ocean_proximity"].unique().tolist(),
            default=df["ocean_proximity"].unique().tolist(),
        )

        income_min, income_max = st.slider(
            "Median Income",
            float(df["median_income"].min()),
            float(df["median_income"].max()),
            (
                float(df["median_income"].min()),
                float(df["median_income"].max()),
            ),
        )

    filtered = df[
        df["ocean_proximity"].isin(prox)
        & df["median_income"].between(income_min, income_max)
    ]

    st.caption(f"Showing **{len(filtered):,}** of **{len(df):,}** records")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📈 Distributions", "🔗 Correlation", "🗺️ Geography", "⚠️ Outliers"]
    )

    with tab1:
        c1, c2 = st.columns(2)

        with c1:
            fig = px.histogram(
                filtered,
                x="median_house_value",
                nbins=60,
                title="Median House Value Distribution",
            )
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig = px.histogram(
                filtered,
                x="median_income",
                nbins=50,
                title="Median Income Distribution",
            )
            st.plotly_chart(fig, use_container_width=True)

        fig = px.box(
            filtered,
            x="ocean_proximity",
            y="median_house_value",
            color="ocean_proximity",
            title="House Value by Ocean Proximity",
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        corr = filtered.select_dtypes(include="number").corr()

        fig = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Feature Correlation Heatmap",
        )
        fig.update_layout(height=650)
        st.plotly_chart(fig, use_container_width=True)

        tc = (
            corr["median_house_value"]
            .drop("median_house_value")
            .sort_values(ascending=False)
            .to_frame("Correlation")
        )

        st.markdown("**Target correlation ranking:**")
        st.dataframe(tc.round(3), use_container_width=True)

    with tab3:
        sample = filtered.sample(
            min(5000, len(filtered)),
            random_state=42,
        )

        fig = px.scatter(
            sample,
            x="longitude",
            y="latitude",
            color="median_house_value",
            size="population",
            hover_data=["median_income", "ocean_proximity"],
            color_continuous_scale="Viridis",
            title="Geographic Distribution of House Values",
        )
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.markdown("### 🚨 Target Cap Analysis")

        capped = int((df["median_house_value"] >= 500000).sum())
        pct = capped / len(df) * 100
        exact_cap = int((df["median_house_value"] == 500001).sum())

        c1, c2, c3 = st.columns(3)
        c1.metric("Values ≥ $500K", f"{capped:,}")
        c2.metric("Percentage", f"{pct:.2f}%")
        c3.metric("Exact $500,001", f"{exact_cap:,}")

        st.info(
            "The target contains a strong concentration at $500,001, "
            "suggesting an upper-value cap in the recorded target."
        )


# =============================================================================
# PAGE 3 — MODEL COMPARISON
# =============================================================================

def page_models():
    st.markdown('<div class="section-title">🤖 Model Comparison</div>',
                unsafe_allow_html=True)

    model_comparison = pd.DataFrame(
        [
            ("Tuned XGBoost", 44409.81, 28872.71, 0.8495),
            ("XGBoost", 46613.96, 30728.51, 0.8342),
            ("Tuned Random Forest", 49526.96, 31792.44, 0.8128),
            ("Random Forest", 49836.04, 31900.63, 0.8105),
            ("KNN", 59709.28, 39340.75, 0.7279),
            ("Tuned Decision Tree", 60184.15, 39902.25, 0.7236),
            ("Linear Regression", 68057.87, 48998.52, 0.6465),
            ("Lasso", 68058.02, 48998.62, 0.6465),
            ("Ridge", 68067.67, 49004.94, 0.6464),
            ("Decision Tree", 69955.29, 43145.08, 0.6265),
            ("SVR", 116976.48, 87094.95, -0.0442),
        ],
        columns=["Model", "RMSE", "MAE", "R²"],
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Final RMSE", "$44.4K")
    c2.metric("Final MAE", "$28.9K")
    c3.metric("Final R²", "0.8495")
    c4.metric("Final MAPE", "16.20%")

    st.markdown("<br>", unsafe_allow_html=True)

    fig = px.bar(
        model_comparison.sort_values("RMSE"),
        x="RMSE",
        y="Model",
        orientation="h",
        title="RMSE Comparison — Lower is Better",
        text="RMSE",
    )
    fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.scatter(
        model_comparison,
        x="RMSE",
        y="R²",
        text="Model",
        size=model_comparison["R²"].abs() + 0.2,
        title="RMSE vs R²",
    )
    fig2.update_traces(textposition="top center")
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### 📋 Full Results")
    st.dataframe(
        model_comparison.style.format(
            {"RMSE": "${:,.0f}", "MAE": "${:,.0f}", "R²": "{:.4f}"}
        ),
        use_container_width=True,
    )


# =============================================================================
# PAGE 4 — PREDICTION
# =============================================================================

def page_predict():
    st.markdown(
        '<div class="section-title">🎯 Live House Value Prediction</div>',
        unsafe_allow_html=True,
    )

    pipeline = load_pipeline()

    if pipeline is None:
        st.error(
            "Model artifact not found. Expected: "
            "`models/california_housing_xgb.joblib`"
        )
        return

    st.info(
        "Enter the property and census-tract characteristics below. "
        "The application automatically creates the engineered features "
        "used during model development."
    )

    st.markdown("### 📍 Location")

    c1, c2, c3 = st.columns(3)

    with c1:
        longitude = st.number_input(
            "Longitude",
            min_value=-124.36,
            max_value=-114.31,
            value=-122.23,
            step=0.01,
        )

    with c2:
        latitude = st.number_input(
            "Latitude",
            min_value=32.53,
            max_value=41.95,
            value=37.88,
            step=0.01,
        )

    with c3:
        ocean_proximity = st.selectbox(
            "Ocean Proximity",
            ["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"],
        )

    st.markdown("### 🏠 Housing Characteristics")

    c1, c2, c3 = st.columns(3)

    with c1:
        housing_median_age = st.number_input(
            "Housing Median Age",
            min_value=1.0,
            max_value=52.0,
            value=28.0,
            step=1.0,
        )

    with c2:
        total_rooms = st.number_input(
            "Total Rooms",
            min_value=1.0,
            value=2000.0,
            step=100.0,
        )

    with c3:
        total_bedrooms = st.number_input(
            "Total Bedrooms",
            min_value=1.0,
            value=400.0,
            step=10.0,
        )

    st.markdown("### 👥 Population & Economics")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        population = st.number_input(
            "Population",
            min_value=1.0,
            value=1000.0,
            step=50.0,
        )

    with c2:
        households = st.number_input(
            "Households",
            min_value=1.0,
            value=350.0,
            step=10.0,
        )

    with c3:
        median_income = st.number_input(
            "Median Income",
            min_value=0.49,
            max_value=15.0,
            value=3.87,
            step=0.1,
        )

    with c4:
        st.metric("Model", "Tuned XGBoost")
        st.caption("RMSE ≈ $44.4K")

    if total_bedrooms > total_rooms:
        st.warning(
            "Total bedrooms is greater than total rooms. "
            "Please check the input."
        )

    st.markdown("---")

    if st.button("🚀 Predict House Value", use_container_width=True):
        if total_bedrooms > total_rooms:
            st.error("Prediction stopped: bedrooms cannot exceed total rooms.")
            return

        raw_input = pd.DataFrame(
            [
                {
                    "longitude": longitude,
                    "latitude": latitude,
                    "housing_median_age": housing_median_age,
                    "total_rooms": total_rooms,
                    "total_bedrooms": total_bedrooms,
                    "population": population,
                    "households": households,
                    "median_income": median_income,
                    "ocean_proximity": ocean_proximity,
                }
            ]
        )

        model_input = engineer_features(raw_input)

        try:
            prediction = float(pipeline.predict(model_input)[0])

            st.markdown(
                f"""
                <div class="prediction-box">
                    <div class="prediction-label">Estimated Median House Value</div>
                    <div class="prediction-value">${prediction:,.0f}</div>
                    <div>Model: Tuned XGBoost</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            with st.expander("🔍 View engineered input"):
                st.dataframe(model_input, use_container_width=True)

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")


# =============================================================================
# PAGE 5 — INSIGHTS
# =============================================================================

def page_insights():
    st.markdown('<div class="section-title">💡 Model & Business Insights</div>',
                unsafe_allow_html=True)

    insights = [
        (
            "Income is a major predictive signal",
            "`median_income` showed the strongest univariate linear association "
            "with the target and also plays an important role in the final model."
        ),
        (
            "Geography matters",
            "House values show clear spatial patterns, with many higher-value "
            "observations concentrated in coastal regions."
        ),
        (
            "Multicollinearity exists",
            "`total_rooms`, `total_bedrooms`, and `households` show strong "
            "relationships. VIF analysis confirmed substantial multicollinearity."
        ),
        (
            "Tree ensembles capture non-linear structure",
            "Random Forest and XGBoost substantially improved over linear models."
        ),
        (
            "The target has a cap",
            "A notable concentration at $500,001 can affect prediction errors "
            "and should be considered when interpreting model performance."
        ),
        (
            "MAPE remains an improvement area",
            "The final model achieved 16.20% MAPE, slightly above the 15% "
            "case-study target."
        ),
    ]

    for title, body in insights:
        st.markdown(
            f"""
            <div class="metric-card" style="margin-bottom: 1rem;">
                <div class="metric-label">{title}</div>
                <p style="margin:0.4rem 0 0 0; color:#374151;">{body}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### 🧪 Robustness")

    robustness = pd.DataFrame(
        [
            (21, 44664.35, 0.8516, 16.12),
            (42, 44409.81, 0.8495, 16.20),
            (100, 43071.10, 0.8623, 15.52),
        ],
        columns=["Random State", "RMSE", "R²", "MAPE %"],
    )

    st.dataframe(
        robustness.style.format(
            {"RMSE": "${:,.0f}", "R²": "{:.4f}", "MAPE %": "{:.2f}%"}
        ),
        use_container_width=True,
    )

    st.info(
        "The robustness experiment uses three train/test random states. "
        "It indicates relatively stable performance across the tested splits, "
        "but it is not proof of universal stability."
    )


# =============================================================================
# ROUTING
# =============================================================================

df = load_data()

if page == "🏠 Overview":
    page_overview(df)

elif page == "📊 EDA":
    page_eda(df)

elif page == "🤖 Model Comparison":
    page_models()

elif page == "🎯 Predict":
    page_predict()

elif page == "💡 Insights":
    page_insights()
'''

out = Path("/mnt/data/app_fixed.py")
out.write_text(app_code, encoding="utf-8")
print(f"Created: {out}")
print(f"Lines: {len(app_code.splitlines())}")
