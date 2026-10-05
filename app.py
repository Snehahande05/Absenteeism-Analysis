from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ---------------------------------------------------------
# Page setup
# ---------------------------------------------------------
st.set_page_config(
    page_title="Employee Absenteeism Analytics",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = Path(__file__).with_name("best_absenteeism_model.pkl")

# ---------------------------------------------------------
# Palette from the supplied theme image
# Buttercup Sky  #FFF2B2
# Dewy Blue      #A8C6E7
# Sunwashed      #FFE08A
# Cloud Puff     #FFF7D6
# Morning Breeze #7FA8D6
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    :root {
        --buttercup: #FFF2B2;
        --dewy-blue: #A8C6E7;
        --sunwashed: #FFE08A;
        --cloud-puff: #FFF7D6;
        --morning-breeze: #7FA8D6;
        --ink: #284B63;
        --ink-dark: #183B56;
        --white: #FFFFFF;
    }

    /* Main page */
    .stApp {
        background: linear-gradient(180deg, #FFF7D6 0%, #FFFDF1 100%);
        color: var(--ink) !important;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Force readable text everywhere */
    .stApp, .stApp p, .stApp span, .stApp label,
    .stApp h1, .stApp h2, .stApp h3, .stApp h4,
    [data-testid="stMarkdownContainer"] {
        color: var(--ink-dark);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #A8C6E7 0%, #FFF2B2 100%);
        border-right: 1px solid #7FA8D6;
    }

    [data-testid="stSidebar"] * {
        color: var(--ink-dark) !important;
    }

    /* Hero */
    .hero {
        background: linear-gradient(120deg, #FFE08A 0%, #FFF2B2 58%, #A8C6E7 100%);
        border: 1px solid #7FA8D6;
        border-radius: 24px;
        padding: 1.7rem 1.9rem;
        box-shadow: 0 10px 28px rgba(127, 168, 214, 0.22);
        margin-bottom: 1.2rem;
    }

    .hero h1 {
        color: var(--ink-dark) !important;
        margin: 0;
        font-size: 2.25rem;
        line-height: 1.15;
    }

    .hero p {
        color: var(--ink) !important;
        margin: .6rem 0 0 0;
        font-size: 1rem;
    }

    /* Statistic cards */
    .stat-card {
        background: rgba(255,255,255,.82);
        border: 1px solid #A8C6E7;
        border-top: 6px solid #FFE08A;
        border-radius: 18px;
        padding: 1rem 1.1rem;
        min-height: 108px;
        box-shadow: 0 7px 20px rgba(127,168,214,.12);
    }

    .stat-label {
        color: #56738D !important;
        font-size: .84rem;
        margin-bottom: .42rem;
        font-weight: 600;
    }

    .stat-value {
        color: var(--ink-dark) !important;
        font-size: 1.65rem;
        font-weight: 800;
    }

    /* Section headings */
    .section-title {
        color: var(--ink-dark) !important;
        font-size: 1.25rem;
        font-weight: 800;
        margin: .45rem 0 .8rem 0;
    }

    .section-note {
        background: #FFF2B2;
        color: var(--ink-dark) !important;
        border-left: 6px solid #7FA8D6;
        border-radius: 13px;
        padding: .9rem 1rem;
        margin-bottom: 1.1rem;
    }

    /* Tabs */
    [data-baseweb="tab-list"] {
        gap: .4rem;
        background: transparent;
    }

    button[data-baseweb="tab"] {
        background: #FFF2B2 !important;
        border-radius: 12px 12px 0 0 !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }

    button[data-baseweb="tab"] * {
        color: var(--ink-dark) !important;
        font-weight: 700 !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        background: #A8C6E7 !important;
    }

    /* Labels */
    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] span,
    label p {
        color: var(--ink-dark) !important;
        font-weight: 700 !important;
    }

    /* Select boxes */
    div[data-baseweb="select"] > div {
        background: #FFFFFF !important;
        border: 1.5px solid #A8C6E7 !important;
        color: var(--ink-dark) !important;
        border-radius: 10px !important;
        min-height: 42px;
    }

    div[data-baseweb="select"] span,
    div[data-baseweb="select"] svg {
        color: var(--ink-dark) !important;
        fill: var(--ink-dark) !important;
    }

    /* Dropdown menu */
    div[data-baseweb="popover"] ul,
    div[data-baseweb="menu"] {
        background: #FFF7D6 !important;
    }

    div[data-baseweb="popover"] li,
    div[data-baseweb="menu"] li,
    div[data-baseweb="popover"] li span {
        color: var(--ink-dark) !important;
    }

    div[data-baseweb="popover"] li:hover {
        background: #A8C6E7 !important;
    }

    /* Number inputs */
    [data-testid="stNumberInput"] input {
        background: #FFFFFF !important;
        color: var(--ink-dark) !important;
        -webkit-text-fill-color: var(--ink-dark) !important;
        border-color: #A8C6E7 !important;
    }

    [data-testid="stNumberInput"] button {
        background: #FFF2B2 !important;
        color: var(--ink-dark) !important;
        border-color: #A8C6E7 !important;
    }

    [data-testid="stNumberInput"] button svg {
        fill: var(--ink-dark) !important;
    }

    /* Main prediction button */
    div.stButton > button[kind="primary"],
    div.stButton > button {
        width: 100%;
        min-height: 48px;
        border-radius: 13px;
        border: 1px solid #658FBE !important;
        background: #7FA8D6 !important;
        color: #FFFFFF !important;
        font-weight: 800 !important;
        box-shadow: 0 6px 16px rgba(127,168,214,.2);
    }

    div.stButton > button * {
        color: #FFFFFF !important;
    }

    div.stButton > button:hover {
        background: #6F9ACA !important;
        color: #FFFFFF !important;
        border-color: #5E88B7 !important;
    }

    /* Result and workflow cards */
    .result-card {
        background: linear-gradient(120deg, #FFF2B2 0%, #FFF7D6 100%);
        border: 1px solid #A8C6E7;
        border-left: 7px solid #7FA8D6;
        border-radius: 18px;
        padding: 1rem 1.2rem;
        min-height: 108px;
        color: var(--ink-dark) !important;
    }

    .workflow-step {
        background: rgba(255,255,255,.8);
        border: 1px solid #A8C6E7;
        border-top: 6px solid #FFE08A;
        border-radius: 17px;
        padding: 1rem;
        min-height: 165px;
        box-shadow: 0 6px 18px rgba(127,168,214,.10);
    }

    .workflow-step h3, .workflow-step p {
        color: var(--ink-dark) !important;
    }

    /* Metrics */
    [data-testid="stMetric"] {
        background: rgba(255,255,255,.85);
        border: 1px solid #A8C6E7;
        border-radius: 16px;
        padding: .85rem 1rem;
    }

    [data-testid="stMetricLabel"],
    [data-testid="stMetricValue"] {
        color: var(--ink-dark) !important;
    }

    /* Expanders / dataframes / alerts */
    [data-testid="stExpander"] {
        background: rgba(255,255,255,.75);
        border: 1px solid #A8C6E7;
        border-radius: 14px;
    }

    [data-testid="stAlert"] {
        color: var(--ink-dark) !important;
    }

    [data-testid="stAlert"] * {
        color: var(--ink-dark) !important;
    }

    /* Captions and help text */
    [data-testid="stCaptionContainer"] *,
    .stCaption {
        color: #55718A !important;
    }

    /* Hide Streamlit's dark top decoration, keep header transparent */
    [data-testid="stHeader"] {
        background: rgba(255,247,214,.92) !important;
    }

    /* Mobile friendliness */
    @media (max-width: 900px) {
        .hero h1 { font-size: 1.8rem; }
        .block-container { padding-left: 1rem; padding-right: 1rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Model helpers
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def normalize_feature_name(value):
    """Normalize whitespace so minor column-name spacing differences are harmless."""
    return " ".join(str(value).split()).strip()


def align_to_model(raw_row, fitted_model):
    """Create one row using the exact feature names stored in the fitted pipeline."""
    expected_columns = list(getattr(fitted_model, "feature_names_in_", raw_row.keys()))
    normalized_values = {
        normalize_feature_name(key): value for key, value in raw_row.items()
    }

    aligned = {}
    for expected in expected_columns:
        normalized_expected = normalize_feature_name(expected)
        if normalized_expected not in normalized_values:
            raise ValueError(f"Missing value for model feature: {expected!r}")
        aligned[expected] = normalized_values[normalized_expected]

    return pd.DataFrame([aligned], columns=expected_columns)


def prediction_band(hours):
    if hours < 4:
        return "Short predicted absence", "The estimated duration is relatively short."
    if hours < 8:
        return "Moderate predicted absence", "The estimate is close to a standard working day."
    return "Longer predicted absence", "The model estimates a comparatively longer absence duration."


if not MODEL_PATH.exists():
    st.error("Model file not found. Keep 'best_absenteeism_model.pkl' in the same folder as app.py.")
    st.stop()

try:
    model = load_model()
except Exception as exc:
    st.error("The trained model could not be loaded.")
    st.code(str(exc))
    st.info("Install the project requirements and make sure scikit-learn 1.6.1 is being used.")
    st.stop()

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("## ☀️ Absenteeism ML")
    st.caption("Employee Absenteeism Analysis & Prediction")
    st.divider()

    st.markdown("### Project Overview")
    st.write("**Task:** Supervised Regression")
    st.write("**Selected Model:** Random Forest")
    st.write("**Target:** Absenteeism time in hours")

    st.divider()
    st.markdown("### Prediction Pipeline")
    st.write("**19 input features**")
    st.write("↓")
    st.write("Encoding + Scaling")
    st.write("↓")
    st.write("Random Forest")
    st.write("↓")
    st.write("Predicted hours")

    st.divider()
    st.caption("Academic demonstration using the Absenteeism at Work dataset.")

# ---------------------------------------------------------
# Header and summary cards
# ---------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>Employee Absenteeism Analytics</h1>
        <p>Interactive machine-learning dashboard for estimating employee absenteeism duration.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown('<div class="stat-card"><div class="stat-label">Dataset Records</div><div class="stat-value">740</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="stat-card"><div class="stat-label">Original Variables</div><div class="stat-value">21</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="stat-card"><div class="stat-label">Model Inputs</div><div class="stat-value">19</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown('<div class="stat-card"><div class="stat-label">Selected Model</div><div class="stat-value">Random Forest</div></div>', unsafe_allow_html=True)

st.write("")
prediction_tab, how_tab, project_tab = st.tabs(["🔮 Prediction", "🧠 How It Works", "📘 Project Details"])

# ---------------------------------------------------------
# Prediction tab
# ---------------------------------------------------------
with prediction_tab:
    st.markdown(
        '<div class="section-note">Enter the employee, absence and workplace information below, then click <b>Predict Absenteeism Duration</b>.</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">1. Absence & Time Information</div>', unsafe_allow_html=True)
    a1, a2, a3, a4 = st.columns(4)

    reason = a1.selectbox(
        "Reason for absence (code)",
        list(range(0, 29)),
        index=1,
        help="The dataset stores reason for absence as a numerical category code.",
    )
    month = a2.selectbox(
        "Month",
        list(range(0, 13)),
        index=1,
        format_func=lambda x: {
            0: "Not specified", 1: "January", 2: "February", 3: "March",
            4: "April", 5: "May", 6: "June", 7: "July", 8: "August",
            9: "September", 10: "October", 11: "November", 12: "December",
        }[x],
    )
    day = a3.selectbox(
        "Day of week",
        [2, 3, 4, 5, 6],
        format_func=lambda x: {2: "Monday", 3: "Tuesday", 4: "Wednesday", 5: "Thursday", 6: "Friday"}[x],
    )
    season = a4.selectbox("Season (code)", [1, 2, 3, 4], format_func=lambda x: f"Season {x}")

    st.markdown('<div class="section-title">2. Employee Profile</div>', unsafe_allow_html=True)
    e1, e2, e3, e4 = st.columns(4)

    age = e1.number_input("Age", min_value=18, max_value=80, value=36)
    service_time = e1.number_input("Service time", min_value=0, max_value=60, value=13)

    education = e2.selectbox("Education (code)", [1, 2, 3, 4], format_func=lambda x: f"Education level {x}")
    son = e2.number_input("Number of children (Son)", min_value=0, max_value=20, value=1)

    social_drinker = e3.selectbox("Social drinker", [0, 1], format_func=lambda x: "Yes" if x else "No")
    social_smoker = e3.selectbox("Social smoker", [0, 1], format_func=lambda x: "Yes" if x else "No")

    pet = e4.number_input("Number of pets", min_value=0, max_value=20, value=0)
    disciplinary = e4.selectbox("Disciplinary failure", [0, 1], format_func=lambda x: "Yes" if x else "No")

    st.markdown('<div class="section-title">3. Physical & Workplace Information</div>', unsafe_allow_html=True)
    p1, p2, p3, p4 = st.columns(4)

    weight = p1.number_input("Weight", min_value=30, max_value=200, value=79)
    height = p1.number_input("Height", min_value=120, max_value=220, value=172)

    bmi = p2.number_input("Body mass index", min_value=10, max_value=60, value=26)
    transport = p2.number_input("Transportation expense", min_value=0, value=239)

    distance = p3.number_input("Distance from Residence to Work", min_value=0, value=29)
    workload = p3.number_input("Work load Average/day", min_value=0.0, value=270.0, step=1.0)

    hit_target = p4.number_input("Hit target", min_value=0, max_value=100, value=95)

    st.write("")
    predict_clicked = st.button("🔮 Predict Absenteeism Duration", type="primary", use_container_width=True)

    if predict_clicked:
        raw_row = {
            "Reason for absence": reason,
            "Month of absence": month,
            "Day of the week": day,
            "Seasons": season,
            "Transportation expense": transport,
            "Distance from Residence to Work": distance,
            "Service time": service_time,
            "Age": age,
            "Work load Average/day": workload,
            "Hit target": hit_target,
            "Disciplinary failure": disciplinary,
            "Education": education,
            "Son": son,
            "Social drinker": social_drinker,
            "Social smoker": social_smoker,
            "Pet": pet,
            "Weight": weight,
            "Height": height,
            "Body mass index": bmi,
        }

        try:
            input_df = align_to_model(raw_row, model)
            raw_prediction = float(model.predict(input_df)[0])
            prediction = max(0.0, raw_prediction)
            band_title, band_text = prediction_band(prediction)

            st.divider()
            st.markdown('<div class="section-title">Prediction Result</div>', unsafe_allow_html=True)
            r1, r2, r3 = st.columns([1, 1, 2])
            r1.metric("Predicted Absenteeism", f"{prediction:.2f} hours")
            r2.metric("Approx. 8-hour Workdays", f"{prediction / 8:.2f} days")
            with r3:
                st.markdown(
                    f'<div class="result-card"><b>{band_title}</b><br><br>{band_text}</div>',
                    unsafe_allow_html=True,
                )

            with st.expander("View submitted input summary"):
                summary_df = pd.DataFrame({"Feature": list(raw_row.keys()), "Entered Value": list(raw_row.values())})
                st.dataframe(summary_df, use_container_width=True, hide_index=True)

            st.caption(
                "The result is a machine-learning estimate based on historical patterns. "
                "It is not a guaranteed outcome and should not be interpreted as a causal conclusion."
            )
        except Exception as exc:
            st.error(f"Prediction could not be generated: {exc}")

# ---------------------------------------------------------
# How it works tab
# ---------------------------------------------------------
with how_tab:
    st.markdown("## How the Complete ML Solution Works")
    st.write(
        "The application uses the same fitted preprocessing pipeline and trained model created during the notebook workflow."
    )

    h1, h2, h3, h4 = st.columns(4)
    with h1:
        st.markdown('<div class="workflow-step"><h3>1️⃣ User Input</h3><p>Collects 19 employee, temporal, physical and workplace-related values.</p></div>', unsafe_allow_html=True)
    with h2:
        st.markdown('<div class="workflow-step"><h3>2️⃣ Feature Preparation</h3><p>Builds one input row using the exact feature names expected by the trained pipeline.</p></div>', unsafe_allow_html=True)
    with h3:
        st.markdown('<div class="workflow-step"><h3>3️⃣ Preprocessing</h3><p>Categorical features are one-hot encoded and numerical features are standardized.</p></div>', unsafe_allow_html=True)
    with h4:
        st.markdown('<div class="workflow-step"><h3>4️⃣ Prediction</h3><p>The selected Random Forest model estimates absenteeism duration in hours.</p></div>', unsafe_allow_html=True)

    st.write("")
    st.markdown("### End-to-End Workflow")
    st.code(
        "User Input\n"
        "   ↓\n"
        "Feature Preparation\n"
        "   ↓\n"
        "One-Hot Encoding + Standard Scaling\n"
        "   ↓\n"
        "Random Forest Regressor\n"
        "   ↓\n"
        "Predicted Absenteeism Hours",
        language="text",
    )

    st.markdown("### Why Random Forest Was Used")
    st.write(
        "Linear Regression, Decision Tree Regressor and Random Forest Regressor were compared using MAE, RMSE and R². "
        "The notebook selected the model with the lowest test-set MAE. In the completed run, Random Forest was selected."
    )

    st.markdown("### Evaluation Measures")
    q1, q2, q3 = st.columns(3)
    q1.info("**MAE**\n\nAverage absolute prediction error in hours. Lower is better.")
    q2.info("**RMSE**\n\nGives greater penalty to large prediction errors. Lower is better.")
    q3.info("**R² Score**\n\nMeasures how much variation in the target is explained by the model.")

    st.markdown("### Important Limitation")
    st.warning(
        "The dataset has 740 records and absenteeism duration is strongly right-skewed. "
        "Very long absences can be harder to predict. The model learns statistical associations from historical data and does not establish causation."
    )

# ---------------------------------------------------------
# Project details tab
# ---------------------------------------------------------
with project_tab:
    st.markdown("## Project Details")

    p1, p2 = st.columns(2)
    with p1:
        st.markdown("### Problem")
        st.write(
            "An employer wants to understand patterns associated with employee absence and estimate absenteeism duration in hours."
        )
        st.markdown("### ML Problem Type")
        st.write("Supervised regression, because the target variable is numerical and measured in hours.")

    with p2:
        st.markdown("### Dataset")
        st.write("Absenteeism at Work — 740 records and 21 original variables.")
        st.markdown("### Target")
        st.write("Absenteeism time in hours")

    st.markdown("### Model Development")
    st.write(
        "Three regression models were compared: Linear Regression, Decision Tree Regressor and Random Forest Regressor. "
        "The final model was selected from actual test-set results."
    )

    st.markdown("### Application Purpose")
    st.write(
        "The Streamlit interface demonstrates the complete solution by collecting new values, "
        "sending them through the fitted preprocessing pipeline and displaying the model prediction."
    )

st.divider()
st.caption("Employee Absenteeism Analysis & Prediction • Streamlit Machine-Learning Application")
