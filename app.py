# =============================================================================
#  California Housing Value Predictor
#  End-to-End Data Science Case Study — Streamlit Dashboard
# -----------------------------------------------------------------------------
#  Author   : <Your Name>
#  Dataset  : California Housing Prices (Kaggle)
#  Best Model: XGBoost (R² = 0.834, RMSE = $46.6K)
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import joblib

# =============================================================================
#  PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="California Housing Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
#  CUSTOM CSS  (Professional Theme)
# =============================================================================
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf1 100%);
    }

    .hero {
        background: linear-gradient(120deg, #1e3c72 0%, #2a5298 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px rgba(30, 60, 114, 0.3);
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
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        height: 100%;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.12);
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
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.4);
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

    .insight-card {
        background: white;
        padding: 1.2rem 1.4rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
        border-left: 5px solid #667eea;
        margin-bottom: 1rem;
    }
    .insight-card h4 {
        margin: 0 0 0.5rem 0;
        color: #1e3c72;
        font-size: 1.1rem;
    }
    .insight-card p {
        margin: 0;
        color: #374151;
        font-size: 0.95rem;
    }

    .stButton > button {
        background: linear-gradient(120deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border: none;
        padding: 0.7rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.2s ease;
        width: 100%;
    }
    .stButton > button:hover {
        background: linear-gradient(120deg, #2a5298 0%, #1e3c72 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(30,60,114,0.3);
        color: white;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%);
    }
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    [data-testid="stSidebar"] .stRadio label {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
#  CONSTANTS & DATA LOADERS
# =============================================================================
DATA_PATH = Path("data/housing.csv")
PREPROCESSOR_PATH = Path("models/preprocessor.pkl")
MODEL_PATH = Path("models/xgb_model.pkl")


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    """Load the housing dataset."""
    if not DATA_PATH.exists():
        return None
    return pd.read_csv(DATA_PATH)


@st.cache_resource(show_spinner=False)
def load_model():
    """Load preprocessor and trained XGBoost model."""
    if not PREPROCESSOR_PATH.exists() or not MODEL_PATH.exists():
        return None, None
    return joblib.load(PREPROCESSOR_PATH), joblib.load(MODEL_PATH)


# Static model comparison data (from your case study)
MODEL_COMPARISON = pd.DataFrame([
    ("XGBoost",              46613.96, 30728.51,  0.8342, "🥇 Best"),
    ("Tuned Random Forest",  49526.96, 31792.44,  0.8128, ""),
    ("Random Forest",        49836.04, 31900.63,  0.8105, ""),
    ("KNN",                  59709.28, 39340.75,  0.7279, ""),
    ("Tuned Decision Tree",  60184.15, 39902.25,  0.7236, ""),
    ("Linear Regression",    68057.87, 48998.52,  0.6465, ""),
    ("Lasso",                68058.02, 48998.62,  0.6465, ""),
    ("Ridge",                68067.67, 49004.94,  0.6464, ""),
    ("Decision Tree",        69955.29, 43145.08,  0.6265, ""),
    ("SVR",                 116976.48, 87094.95, -0.0442, "⚠️ Poor"),
], columns=["Model", "RMSE", "MAE", "R²", "Status"]).sort_values("RMSE")


FEATURE_IMPORTANCE = pd.DataFrame({
    "Feature": [
        "median_income", "ocean_proximity_INLAND",
        "population_per_household", "latitude", "longitude",
        "housing_median_age", "rooms_per_household",
        "bedrooms_per_room", "total_rooms",
    ],
    "Importance": [
        0.4827, 0.1390, 0.1209, 0.0547, 0.0545,
        0.0440, 0.0254, 0.0239, 0.0125,
    ],
}).sort_values("Importance")


# =============================================================================
#  HERO HEADER
# =============================================================================
st.markdown("""
<div class="hero">
    <h1>🏠 California Housing Value Predictor</h1>
    <p>End-to-End Data Science Case Study — Predicting Median House Values using Census Tract Data</p>
</div>
""", unsafe_allow_html=True)


# =============================================================================
#  SIDEBAR NAVIGATION
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
    st.markdown("""
    - **Dataset:** 20,640 rows × 10 cols
    - **Target:** `median_house_value`
    - **Best Model:** XGBoost
    - **R²:** 0.834
    - **RMSE:** $46.6K
    """)
    st.markdown("---")
    st.caption("Built with Streamlit • XGBoost • Plotly")


# =============================================================================
#  PAGE 1 — OVERVIEW
# =============================================================================
def page_overview(df):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""<div class="metric-card">
            <div class="metric-label">Best Model</div>
            <div class="metric-value">XGBoost</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="metric-card">
            <div class="metric-label">Test R²</div>
            <div class="metric-value">0.834</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="metric-card">
            <div class="metric-label">RMSE</div>
            <div class="metric-value">$46.6K</div></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown("""<div class="metric-card">
            <div class="metric-label">MAE</div>
            <div class="metric-value">$30.7K</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns([2, 1])
    with left:
        st.markdown('<div class="section-title">📌 Project Overview</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        This dashboard demonstrates a complete data-science workflow on the
        **California Housing Prices** dataset:

        - 🧹 **Data Quality Audit** — missing values, duplicates, IQR outliers
        - 📊 **Exploratory Data Analysis** — distributions, correlations, geography
        - 🛠️ **Feature Engineering** — `rooms_per_household`, `bedrooms_per_room`, `population_per_household`
        - 🤖 **Model Comparison** — from Linear Regression to XGBoost
        - 🔍 **Residual Diagnostics** — error analysis around the $500K cap
        - 🎯 **Live Prediction** — enter features, get instant estimate

        Use the sidebar to explore each module.
        """)

    with right:
        st.markdown('<div class="section-title">📊 Dataset Snapshot</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        | Attribute | Value |
        |-----------|-------|
        | Rows | 20,640 |
        | Features | 10 |
        | Target | `median_house_value` |
        | Missing | `total_bedrooms` (~1%) |
        | Categories | 5 (`ocean_proximity`) |
        """)

    if df is not None:
        st.markdown('<div class="section-title">👀 Data Preview</div>',
                    unsafe_allow_html=True)
        st.dataframe(df.head(10), use_container_width=True)
    else:
        st.warning("⚠️ `data/housing.csv` not found. Add it to the `data/` folder.")


# =============================================================================
#  PAGE 2 — EDA
# =============================================================================
def page_eda(df):
    if df is None:
        st.warning("⚠️ `data/housing.csv` not found.")
        return

    st.markdown('<div class="section-title">📊 Exploratory Data Analysis</div>',
                unsafe_allow_html=True)

    # Sidebar filters
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
            (float(df["median_income"].min()), float(df["median_income"].max())),
        )

    filtered = df[
        (df["ocean_proximity"].isin(prox)) &
        (df["median_income"].between(income_min, income_max))
    ]
    st.caption(f"Showing **{len(filtered):,}** of **{len(df):,}** records")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📈 Distributions", "🔗 Correlation", "🗺️ Geography", "⚠️ Outliers"]
    )

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(
                filtered, x="median_house_value", nbins=60,
                title="Median House Value Distribution",
                color_discrete_sequence=["#2a5298"],
            )
            fig.update_layout(bargap=0.05)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.histogram(
                filtered, x="median_income", nbins=50,
                title="Median Income Distribution",
                color_discrete_sequence=["#667eea"],
            )
            st.plotly_chart(fig, use_container_width=True)

        fig = px.box(
            filtered, x="ocean_proximity", y="median_house_value",
            color="ocean_proximity",
            title="House Value by Ocean Proximity",
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        corr = filtered.select_dtypes("number").corr()
        fig = px.imshow(
            corr, text_auto=".2f", aspect="auto",
            color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            title="Feature Correlation Heatmap",
        )
        fig.update_layout(height=650)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Top correlations with target:**")
        tc = corr["median_house_value"].drop("median_house_value").sort_values(ascending=False)
        st.dataframe(tc.to_frame("Correlation").round(3), use_container_width=True)

    with tab3:
        sample = filtered.sample(min(5000, len(filtered)), random_state=42)
        fig = px.scatter_mapbox(
            sample, lat="latitude", lon="longitude",
            color="median_house_value", size="population",
            color_continuous_scale="Viridis",
            zoom=4, height=650,
            mapbox_style="carto-positron",
            title="Geographic Distribution of House Values",
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.markdown("### 🚨 The $500,001 Cap")
        capped = (df["median_house_value"] >= 500000).sum()
        pct = capped / len(df) * 100
        c1, c2, c3 = st.columns(3)
        c1.metric("Capped Values", f"{capped:,}")
        c2.metric("Percentage", f"{pct:.2f}%")
        c3.metric("Exact $500,001", f"{(df['median_house_value']==500001).sum():,}")
        st.info("The target variable is capped at $500,001 — a known data limitation.")


# =============================================================================
#  PAGE 3 — MODEL COMPARISON
# =============================================================================
def page_models():
    st.markdown('<div class="section-title">🤖 Model Comparison</div>',
                unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class="metric-card">
            <div class="metric-label">🥇 Best RMSE</div>
            <div class="metric-value">$46.6K</div>
            <p style="margin:0; color:#6b7280; font-size:0.85rem;">XGBoost</p>
            </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="metric-card">
            <div class="metric-label">🎯 Best R²</div>
            <div class="metric-value">0.834</div>
            <p style="margin:0; color:#6b7280; font-size:0.85rem;">XGBoost</p>
            </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="metric-card">
            <div class="metric-label">⚠️ MAPE</div>
            <div class="metric-value">17.31%</div>
            <p style="margin:0; color:#6b7280; font-size:0.85rem;">Above 15% target</p>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    fig = px.bar(
        MODEL_COMPARISON, x="RMSE", y="Model", orientation="h",
        color="RMSE", color_continuous_scale="Blues_r",
        title="RMSE Comparison (lower is better)",
    )
    fig.update_layout(yaxis=dict(autorange="reversed"), height=500, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.scatter(
        MODEL_COMPARISON, x="RMSE", y="R²", text="Model",
        color="R²", color_continuous_scale="Viridis",
        size=MODEL_COMPARISON["R²"].abs() + 0.2,
        title="RMSE vs R² Trade-off",
    )
    fig2.update_traces(textposition="top center")
    fig2.update_layout(height=550)
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### 📋 Full Results Table")
    st.dataframe(
        MODEL_COMPARISON.style
            .format({"RMSE": "${:,.0f}", "MAE": "${:,.0f}", "R²": "{:.4f}"})
            .background_gradient(subset=["RMSE"], cmap="Blues_r")
            .background_gradient(subset=["R²"], cmap="Greens"),
        use_container_width=True,
    )

    st.success("""
    **Key Takeaways**
    - Tree-based ensembles (XGBoost, Random Forest) far outperform linear models.
    - XGBoost achieves ~$46.6K RMSE and R² = 0.834.
    - SVR underperforms due to scale/parameter sensitivity.
    - Linear models plateau at R² ≈ 0.65.
    """)


# =============================================================================
#  PAGE 4 — PREDICTION
# =============================================================================

def page_predict():
    st.markdown(
        '<div class="section-title">🎯 Live House Value Prediction</div>',
        unsafe_allow_html=True
    )

    preprocessor, model = load_model()

    if model is None:
        st.error("⚠️ Model artifact not found in `models/`.")
        st.stop()