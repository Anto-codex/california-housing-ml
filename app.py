# =============================================================================
# California Housing Value Predictor
# End-to-End Data Science Case Study — Streamlit Dashboard
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
import joblib
from textwrap import dedent


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
    dedent("""
    <style>
        /* =========================================================
           THEME-AWARE DASHBOARD STYLING
           Uses Streamlit theme variables so both light and dark
           modes remain readable without changing the layout.
           ========================================================= */

        [data-testid="stAppViewContainer"] {
            background: var(--background-color, #f5f7fa) !important;
            color: var(--text-color, #1f2937) !important;
        }

        [data-testid="stHeader"] {
            background: rgba(0, 0, 0, 0) !important;
        }

        /* ---------------------------------------------------------
           HERO
           --------------------------------------------------------- */

        .hero {
            background: linear-gradient(120deg, #1e3c72 0%, #2a5298 100%);
            padding: 2.4rem 2rem;
            border-radius: 16px;
            color: #ffffff !important;
            margin-bottom: 1.6rem;
            box-shadow: 0 10px 30px rgba(30, 60, 114, 0.25);
        }

        .hero h1 {
            font-size: 2.5rem;
            font-weight: 750;
            margin: 0 0 0.45rem 0;
            color: #ffffff !important;
        }

        .hero p {
            font-size: 1.05rem;
            opacity: 0.95;
            margin: 0;
            color: #f8fafc !important;
        }

        /* ---------------------------------------------------------
           GLOBAL CONTENT
           --------------------------------------------------------- */

        [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] li,
        [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] strong,
        [data-testid="stAppViewContainer"] [data-testid="stCaptionContainer"] {
            color: var(--text-color, #1f2937) !important;
        }

        /* Bright blue headings work on both light and dark themes. */
        [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] h1,
        [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] h2,
        [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] h3,
        [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] h4,
        [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] h5,
        [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] h6 {
            color: #60a5fa !important;
        }

        /* ---------------------------------------------------------
           CUSTOM SECTION TITLES
           --------------------------------------------------------- */

        .section-title {
            font-size: 1.45rem;
            font-weight: 750;
            color: #60a5fa !important;
            border-bottom: 3px solid #60a5fa !important;
            padding-bottom: 0.4rem;
            margin: 1.5rem 0 1rem 0;
            display: inline-block;
        }

        /* ---------------------------------------------------------
           METRIC CARDS
           --------------------------------------------------------- */

        .metric-card {
            background: var(--secondary-background-color, #ffffff);
            padding: 1.3rem 1.45rem;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
            border-left: 5px solid #2a5298;
            height: 100%;
        }

        .metric-card * {
            color: var(--text-color, #1f2937);
        }

        .metric-label {
            font-size: 0.78rem;
            color: var(--text-color, #6b7280) !important;
            text-transform: uppercase;
            font-weight: 700;
            letter-spacing: 0.6px;
            margin-bottom: 0.35rem;
        }

        .metric-value {
            font-size: 1.85rem;
            font-weight: 750;
            color: #60a5fa !important;
            line-height: 1.15;
        }

        /* ---------------------------------------------------------
           INFO CARDS
           --------------------------------------------------------- */

        .info-card {
            background: var(--secondary-background-color, #ffffff);
            padding: 1.25rem 1.4rem;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.07);
            margin-bottom: 1rem;
        }

        .info-title {
            color: #60a5fa !important;
            font-weight: 750;
            font-size: 1rem;
            margin-bottom: 0.35rem;
        }

        .info-body {
            color: var(--text-color, #374151) !important;
            line-height: 1.55;
        }

        /* ---------------------------------------------------------
           PREDICTION
           --------------------------------------------------------- */

        .prediction-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 14px;
            color: #ffffff !important;
            text-align: center;
            box-shadow: 0 10px 25px rgba(102, 126, 234, 0.35);
            margin: 1.2rem 0;
        }

        .prediction-box * {
            color: #ffffff !important;
        }

        .prediction-label {
            font-size: 0.95rem;
            opacity: 0.92;
            text-transform: uppercase;
            letter-spacing: 1.2px;
        }

        .prediction-value {
            font-size: 3rem;
            font-weight: 800;
            margin: 0.45rem 0;
        }

        .prediction-model {
            font-size: 0.95rem;
            opacity: 0.9;
        }

        /* ---------------------------------------------------------
           BUTTON
           --------------------------------------------------------- */

        .stButton > button {
            width: 100%;
            border-radius: 9px;
            font-weight: 700;
        }

        /* ---------------------------------------------------------
           INPUTS / LABELS
           --------------------------------------------------------- */

        [data-testid="stAppViewContainer"] label {
            color: var(--text-color, #374151) !important;
        }

        [data-testid="stAppViewContainer"] input,
        [data-testid="stAppViewContainer"] textarea,
        [data-testid="stAppViewContainer"] [data-baseweb="select"] * {
            color: var(--text-color, #1f2937) !important;
        }

        [data-testid="stAppViewContainer"] input,
        [data-testid="stAppViewContainer"] textarea,
        [data-testid="stAppViewContainer"] [data-baseweb="select"] > div {
            background: var(--secondary-background-color, #ffffff) !important;
            border-color: #64748b !important;
        }

        /* ---------------------------------------------------------
           NATIVE METRICS
           --------------------------------------------------------- */

        [data-testid="stMetric"] {
            background: var(--secondary-background-color, #ffffff) !important;
            color: var(--text-color, #1f2937) !important;
            border-radius: 12px;
        }

        [data-testid="stMetricLabel"],
        [data-testid="stMetricLabel"] p {
            color: var(--text-color, #6b7280) !important;
        }

        [data-testid="stMetricValue"],
        [data-testid="stMetricValue"] div {
            color: #60a5fa !important;
        }

        /* ---------------------------------------------------------
           ALERTS / INFO BOXES
           --------------------------------------------------------- */

        [data-testid="stAlert"] {
            color: var(--text-color, #374151) !important;
        }

        [data-testid="stAlert"] p,
        [data-testid="stAlert"] div {
            color: var(--text-color, #374151) !important;
        }

        /* ---------------------------------------------------------
           TABLES
           --------------------------------------------------------- */

        [data-testid="stAppViewContainer"] table,
        [data-testid="stAppViewContainer"] th,
        [data-testid="stAppViewContainer"] td {
            color: var(--text-color, #1f2937) !important;
        }

        /* ---------------------------------------------------------
           SIDEBAR — keep existing blue/white design unchanged
           --------------------------------------------------------- */

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%);
        }

        [data-testid="stSidebar"] *,
        [data-testid="stSidebar"] .stMarkdown,
        [data-testid="stSidebar"] label {
            color: #ffffff !important;
        }
    </style>
    """),
    unsafe_allow_html=True,
)


# =============================================================================
# PATHS AND MODEL CONSTANTS
# =============================================================================

DATA_PATH = Path("data/housing.csv")
MODEL_PATH = Path("models/california_housing_xgb.joblib")

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
# DATA AND MODEL LOADING
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
    df = input_df.copy()

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
# HEADER
# =============================================================================

st.markdown(
    dedent("""
    <div class="hero">
        <h1>🏠 California Housing Value Predictor</h1>
        <p>
            End-to-End Data Science Case Study —
            Predicting Median House Values
        </p>
    </div>
    """),
    unsafe_allow_html=True,
)


# =============================================================================
# SIDEBAR
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
        - **Original Features:** 10
        - **Target:** `median_house_value`
        - **Final Model:** Tuned XGBoost
        - **R²:** 0.8495
        - **RMSE:** $44.4K
        """
    )

    st.markdown("---")
    st.caption("Built with Streamlit • XGBoost • Plotly")


# =============================================================================
# OVERVIEW
# =============================================================================

def page_overview(df):

    st.markdown(
        '<div class="section-title">📊 Model Performance</div>',
        unsafe_allow_html=True,
    )

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
                dedent(f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                </div>
                """),
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns([2, 1])

    with left:
        st.markdown(
            '<div class="section-title">📌 Project Overview</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            This application presents an end-to-end Machine Learning
            case study for predicting California median house values.

            **Workflow**

            - Data quality auditing
            - Exploratory data analysis
            - Missing value analysis
            - Outlier analysis
            - Feature engineering
            - Categorical encoding
            - Spatial feature engineering
            - Multicollinearity analysis
            - Model comparison
            - Hyperparameter tuning
            - Explainability
            - Robustness analysis
            - Model serialization
            - Batch inference
            - Monitoring and drift analysis
            """
        )

    with right:
        st.markdown(
            '<div class="section-title">🎯 Business Targets</div>',
            unsafe_allow_html=True,
        )

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
            "MAPE is 16.20%, so the MAPE target remains slightly "
            "above the case-study threshold."
        )

    if df is not None:
        st.markdown(
            '<div class="section-title">👀 Data Preview</div>',
            unsafe_allow_html=True,
        )

        st.dataframe(df.head(10), use_container_width=True)


# =============================================================================
# EDA
# =============================================================================

def page_eda(df):

    if df is None:
        st.error("Dataset not found at `data/housing.csv`.")
        return

    st.markdown(
        '<div class="section-title">📊 Exploratory Data Analysis</div>',
        unsafe_allow_html=True,
    )

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

    filtered_df = df[
        df["ocean_proximity"].isin(prox)
        & df["median_income"].between(income_min, income_max)
    ]

    st.write(
        f"Showing **{len(filtered_df):,}** of **{len(df):,}** observations."
    )

    st.markdown("### 📈 Median House Value Distribution")

    fig = px.histogram(
        filtered_df,
        x="median_house_value",
        nbins=50,
        title="Distribution of Median House Value",
    )

    fig.update_layout(
        xaxis_title="Median House Value",
        yaxis_title="Count",
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 💰 Median Income vs House Value")

    fig = px.scatter(
        filtered_df,
        x="median_income",
        y="median_house_value",
        color="ocean_proximity",
        opacity=0.35,
        title="Median Income vs Median House Value",
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🌊 House Value by Ocean Proximity")

    fig = px.box(
        filtered_df,
        x="ocean_proximity",
        y="median_house_value",
        color="ocean_proximity",
        title="Median House Value by Ocean Proximity",
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🗺️ Geographic Distribution")

    fig = px.scatter(
        filtered_df,
        x="longitude",
        y="latitude",
        color="median_house_value",
        size="population",
        hover_data=[
            "median_income",
            "ocean_proximity",
            "housing_median_age",
        ],
        opacity=0.5,
        title="California Housing Values by Geographic Location",
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🔗 Correlation Analysis")

    numeric_df = filtered_df.select_dtypes(include=np.number)
    correlation = numeric_df.corr()

    fig = px.imshow(
        correlation,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Numerical Feature Correlation",
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### ⚠️ Target Cap Investigation")

    capped = int((df["median_house_value"] >= 500000).sum())
    exact_cap = int((df["median_house_value"] == 500001).sum())
    pct = capped / len(df) * 100

    c1, c2, c3 = st.columns(3)

    c1.metric("Values ≥ $500K", f"{capped:,}")
    c2.metric("Percentage", f"{pct:.2f}%")
    c3.metric("Exact $500,001", f"{exact_cap:,}")

    st.info(
        "The target contains a strong concentration at $500,001, "
        "suggesting an upper-value cap in the recorded target."
    )


# =============================================================================
# MODEL COMPARISON
# =============================================================================

def page_models():

    st.markdown(
        '<div class="section-title">🤖 Model Comparison</div>',
        unsafe_allow_html=True,
    )

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

    fig.update_traces(
        texttemplate="$%{text:,.0f}",
        textposition="outside",
    )

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
            {
                "RMSE": "${:,.0f}",
                "MAE": "${:,.0f}",
                "R²": "{:.4f}",
            }
        ),
        use_container_width=True,
    )


# =============================================================================
# PREDICTION
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
            [
                "<1H OCEAN",
                "INLAND",
                "NEAR OCEAN",
                "NEAR BAY",
                "ISLAND",
            ],
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

    if st.button(
        "🚀 Predict House Value",
        use_container_width=True,
        type="primary",
    ):

        if total_bedrooms > total_rooms:
            st.error(
                "Prediction stopped: bedrooms cannot exceed total rooms."
            )
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
                dedent(f"""
                <div class="prediction-box">
                    <div class="prediction-label">
                        Estimated Median House Value
                    </div>
                    <div class="prediction-value">
                        ${prediction:,.0f}
                    </div>
                    <div class="prediction-model">
                        Model: Tuned XGBoost
                    </div>
                </div>
                """),
                unsafe_allow_html=True,
            )

            st.success(
                "Prediction generated successfully."
            )

            with st.expander("🔍 View Engineered Input"):
                st.dataframe(
                    model_input,
                    use_container_width=True,
                )

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")


# =============================================================================
# INSIGHTS
# =============================================================================

def page_insights():

    st.markdown(
        '<div class="section-title">💡 Model & Business Insights</div>',
        unsafe_allow_html=True,
    )

    insights = [
        (
            "Income is a major predictive signal",
            "`median_income` showed the strongest univariate "
            "linear association with the target and also plays "
            "an important role in the final model.",
        ),
        (
            "Geography matters",
            "House values show clear spatial patterns, with many "
            "higher-value observations concentrated in coastal regions.",
        ),
        (
            "Multicollinearity exists",
            "`total_rooms`, `total_bedrooms`, and `households` "
            "show strong relationships. VIF analysis confirmed "
            "substantial multicollinearity.",
        ),
        (
            "Tree ensembles capture non-linear structure",
            "Random Forest and XGBoost substantially improved "
            "over linear models.",
        ),
        (
            "The target has a cap",
            "A notable concentration at $500,001 can affect "
            "prediction errors and should be considered when "
            "interpreting model performance.",
        ),
        (
            "MAPE remains an improvement area",
            "The final model achieved 16.20% MAPE, slightly above "
            "the 15% case-study target.",
        ),
    ]

    for title, body in insights:
        st.markdown(
            dedent(f"""
            <div class="info-card">
                <div class="info-title">{title}</div>
                <div class="info-body">{body}</div>
            </div>
            """),
            unsafe_allow_html=True,
        )

    st.markdown("### 🧪 Robustness Analysis")

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
            {
                "RMSE": "${:,.0f}",
                "R²": "{:.4f}",
                "MAPE %": "{:.2f}%",
            }
        ),
        use_container_width=True,
    )

    st.info(
        "The robustness experiment uses three train/test random states. "
        "It indicates relatively stable performance across the tested "
        "splits, but it is not proof of universal stability."
    )


# =============================================================================
# MAIN ROUTING
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
