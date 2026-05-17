import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Page configurations
st.set_page_config(page_title="Loan Predictor Dashboard", layout="centered")

# --- TRAIN MODEL ON THE FLY ---
# This function reads your CSV directly from your repo and builds the model instantly.
# st.cache_resource ensures this only runs ONCE when the app starts, keeping it lightning fast.
@st.cache_resource
def train_and_prepare_model():
    # 1. Load data (Make sure 'loan_approval.csv' is in your GitHub repo root folder)
    df = pd.read_csv('loan_approval.csv')
    
    # 2. Preprocessing (Matching your notebook exactly)
    df.drop(['city', 'name'], axis=1, inplace=True)
    df['loan_approved'] = df['loan_approved'].astype(int)
    
    # 3. Split Features and Target
    X = df.drop('loan_approved', axis=1)
    y = df['loan_approved']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Scale Features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 5. Train Logistic Regression
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)
    
    # 6. Calculate Accuracy
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    
    return model, scaler, acc

# Initialize the assets
try:
    model, scaler, model_accuracy = train_and_prepare_model()
except FileNotFoundError:
    st.error("⚠️ **Error:** `loan_approval.csv` not found in your repository root directory. Please upload your CSV file to GitHub alongside this `app.py` file.")
    st.stop()


# --- WEB APP UI ---
st.title("🏦 Loan Approval Classification App")
st.write("Provide applicant specifications below to compute the machine learning prediction.")

# Sidebar metrics showing accuracy dynamically
st.sidebar.header("📊 Model Metrics")
st.sidebar.metric(label="Logistic Regression Accuracy", value=f"{model_accuracy * 100:.2f}%")
st.sidebar.write("This model builds dynamically and scales inputs matching your notebook architecture.")


# --- USER INPUT FORM ---
st.subheader("📋 Enter Applicant Profiles")

col1, col2 = st.columns(2)

with col1:
    income = st.number_input("Annual Income ($)", min_value=0, value=50000, step=1000)
    loan_amount = st.number_input("Requested Loan Amount ($)", min_value=0, value=25000, step=1000)
    years_employed = st.number_input("Years of Employment", min_value=0, value=5, step=1)

with col2:
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650, step=10)
    points = st.number_input("Internal Credit Points Assessment", min_value=0.0, max_value=100.0, value=50.0, step=5.0)


# --- PREDICTION LOGIC ---
st.markdown("---")
if st.button("🚀 Predict Loan Status", type="primary"):
    
    # 1. Structure raw inputs into array
    raw_features = np.array([[income, credit_score, loan_amount, years_employed, points]])
    
    # 2. Scale features using the active runtime scaler
    scaled_features = scaler.transform(raw_features)
    
    # 3. Generate predictions
    prediction = model.predict(scaled_features)
    probabilities = model.predict_proba(scaled_features)[0]
    
    # --- DISPLAY OUTPUT RESULTS ---
    if prediction[0] == 1:
        st.success("🎉 **Decision Result: APPROVED**")
        st.info(f"Model Assurance Score: **{probabilities[1] * 100:.2f}%** confidence in approval.")
    else:
        st.error("❌ **Decision Result: REJECTED**")
        st.info(f"Model Assurance Score: **{probabilities[0] * 100:.2f}%** confidence in rejection.")
