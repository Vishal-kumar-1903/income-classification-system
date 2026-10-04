import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Income Level Predictor",
    page_icon="💼",
    layout="wide"
)

st.markdown("""
<style>
    .stApp {
        background-color: #0b1120;
        color: #f8fafc;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #f8fafc !important;
        font-weight: 600 !important;
    }
    
    [data-testid="stSidebar"] {
        background-color: #111827 !important;
        border-right: 1px solid #1f2937;
    }
    
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] span {
        color: #f1f5f9 !important;
        font-weight: 500 !important;
        font-size: 14px !important;
    }
    
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #38bdf8 !important;
        border-bottom: 1px solid #1f2937;
        padding-bottom: 6px;
        margin-top: 10px;
    }
    
    [data-testid="stSidebar"] div[data-baseweb="select"] > div,
    [data-testid="stSidebar"] div[data-baseweb="input"] > div {
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
        color: #ffffff !important;
    }
    
    .metric-card {
        background: #1e293b;
        border-radius: 10px;
        padding: 20px;
        border: 1px solid #334155;
        margin-bottom: 15px;
    }
    .high-income {
        border-left: 5px solid #22c55e;
    }
    .standard-income {
        border-left: 5px solid #38bdf8;
    }
    .footer-text {
        text-align: center;
        color: #94a3b8;
        font-size: 13px;
        margin-top: 40px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_assets():
    model = joblib.load("income_dt_model.pkl")
    encoders = joblib.load("encoders.pkl")
    features = joblib.load("feature_names.pkl")
    return model, encoders, features

try:
    model, encoders, feature_names = load_model_assets()
except Exception:
    st.error("Please run 'python train.py' first to train models!")
    st.stop()

def get_default_index(options_list, preferred_val):
    options = list(options_list)
    return options.index(preferred_val) if preferred_val in options else 0

st.title("💼 Income Category Prediction System")
st.write("An interactive machine learning tool to classify candidate income brackets using demographic and employment data.")

# Sidebar Controls
st.sidebar.header("Candidate Information")

age = st.sidebar.slider("Age", 18, 75, 24)
gender = st.sidebar.radio("Gender", encoders["sex"].classes_, horizontal=True)

edu_options = list(encoders["education"].classes_)
education = st.sidebar.selectbox("Highest Education", edu_options, index=get_default_index(edu_options, "Bachelors"))

education_years = st.sidebar.slider("Total Education Years", 1, 16, 14)

work_options = list(encoders["workclass"].classes_)
work_type = st.sidebar.selectbox("Employment Sector", work_options, index=get_default_index(work_options, "Private"))

occ_options = list(encoders["occupation"].classes_)
occupation = st.sidebar.selectbox("Job Role / Profession", occ_options, index=get_default_index(occ_options, "Tech-support"))

marital_options = list(encoders["marital_status"].classes_)
marital_status = st.sidebar.selectbox("Marital Status", marital_options, index=get_default_index(marital_options, "Never-married"))

st.sidebar.subheader("Work & Financial Details")
daily_hours = st.sidebar.slider("Daily Working Hours (hrs/day)", 2, 14, 8)
hours_per_week = daily_hours * 5

# Extra Profit / Gains options rakha hai simplified language mein
capital_gain = st.sidebar.number_input("Annual Investment / Asset Profit ($)", min_value=0, value=0, step=500, help="Profit from shares, properties, or side investments (Default is 0)")
capital_loss = st.sidebar.number_input("Annual Investment Loss ($)", min_value=0, value=0, step=100, help="Loss from trading or investments (Default is 0)")

user_input = {
    "age": age,
    "workclass": encoders["workclass"].transform([work_type])[0],
    "education": encoders["education"].transform([education])[0],
    "education_num": education_years,
    "marital_status": encoders["marital_status"].transform([marital_status])[0],
    "occupation": encoders["occupation"].transform([occupation])[0],
    "sex": encoders["sex"].transform([gender])[0],
    "capital_gain": capital_gain,
    "capital_loss": capital_loss,
    "hours_per_week": hours_per_week
}

input_df = pd.DataFrame([user_input])[feature_names]

# Main Dashboard Layout
tab1, tab2 = st.tabs(["🔍 Predict Income", "📈 Model Insights & Feature Weights"])

with tab1:
    col1, col2 = st.columns([1, 1.2], gap="large")

    with col1:
        st.subheader("Selected Candidate Summary")
        summary_table = pd.DataFrame({
            "Parameter": ["Age", "Gender", "Education", "Education Years", "Sector", "Profession", "Marital Status", "Daily Shift", "Investment Profit"],
            "Details": [f"{age} years", gender, education, f"{education_years} yrs", work_type, occupation, marital_status, f"{daily_hours} hrs/day", f"${capital_gain}"]
        })
        st.dataframe(summary_table, use_container_width=True, hide_index=True)

    with col2:
        st.subheader("Prediction Result")
        if st.button("Check Income Category", type="primary", use_container_width=True):
            prediction = model.predict(input_df)[0]
            confidence = model.predict_proba(input_df)[0]

            high_income_prob = confidence[1] * 100
            st.write(f"**High Income Confidence (> $50K):** `{high_income_prob:.1f}%`")
            st.progress(float(confidence[1]))

            if prediction == 1:
                st.markdown(f"""
                <div class="metric-card high-income">
                    <h3>🟢 Category: High Income Bracket (> $50K / Year)</h3>
                    <p>Candidate profile, education tier, and work hours indicate higher earning potential.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="metric-card standard-income">
                    <h3>🔵 Category: Standard Income Bracket (≤ $50K / Year)</h3>
                    <p>Candidate profile falls within the standard census earning threshold.</p>
                </div>
                """, unsafe_allow_html=True)

with tab2:
    st.subheader("Top Contributing Features (Decision Tree)")
    st.write("Factors that influence the model's income prediction:")
    feature_importance = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=True)
    st.bar_chart(feature_importance)

    st.subheader("Algorithm Benchmarking")
    comparison_data = pd.DataFrame({
        "Model": ["Logistic Regression", "Decision Tree Classifier (Active)"],
        "Test Accuracy": ["~82.4%", "~85.6%"],
        "Strength": ["Fast baseline, linear boundary", "Handles complex feature interactions better"]
    })
    st.table(comparison_data)

st.markdown("""
<div class="footer-text">
    Minor Project | Machine Learning Income Classification Engine | Built with Scikit-Learn & Streamlit
</div>
""", unsafe_allow_html=True)
