import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Page Configuration
st.set_page_config(page_title="Loan Predictor Dashboard", layout="centered")



# 3. Model Pipeline Runtime (Cached for performance)
@st.cache_resource
def train_and_prepare_model():
    # Load dataset directly from your GitHub repo root
    df = pd.read_csv('loan_approval.csv')
    
    # Preprocessing matching your notebook exactly
    df.drop(['city', 'name'], axis=1, inplace=True)
    df['loan_approved'] = df['loan_approved'].astype(int)
    
    # Split Features and Target
    X = df.drop('loan_approved', axis=1)
    y = df['loan_approved']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale Features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Logistic Regression Model
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)
    
    # Evaluate Accuracy
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    
    return model, scaler, acc

# Load pipeline assets
try:
    model, scaler, model_accuracy = train_and_prepare_model()
except FileNotFoundError:
    st.error("⚠️ **File Missing:** Please make sure `loan_approval.csv` is uploaded to the root of your GitHub repository next to this `app.py` file.")
    st.stop()

# 4. User Interface Content
st.title("🏦 Loan Approval Classification App")
st.write("Provide applicant specifications below to compute the machine learning prediction.")

# Sidebar Metrics Section
st.sidebar.header("📊 Model Metrics")
st.sidebar.metric(label="Logistic Regression Accuracy", value=f"{model_accuracy * 100:.2f}%")
st.sidebar.write("This model builds dynamically and scales inputs matching your notebook architecture.")

# Input Layout
st.subheader("📋 Enter Applicant Profiles")
col1, col2 = st.columns(2)

with col1:
    income = st.number_input("Annual Income ($)", min_value=0, value=50000, step=1000)
    loan_amount = st.number_input("Requested Loan Amount ($)", min_value=0, value=25000, step=1000)
    years_employed = st.number_input("Years of Employment", min_value=0, value=5, step=1)

with col2:
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650, step=10)
    points = st.number_input("Internal Credit Points Assessment", min_value=0.0, max_value=100.0, value=50.0, step=5.0)

st.markdown("---")

# 5. Prediction Logic Execution
if st.button("🚀 Predict Loan Status"):
    # Convert inputs to Numpy array shape
    raw_features = np.array([[income, credit_score, loan_amount, years_employed, points]])
    
    # Run features through the active scaler instance
    scaled_features = scaler.transform(raw_features)
    
    # Compute Classification Results
    prediction = model.predict(scaled_features)
    probabilities = model.predict_proba(scaled_features)[0]
    
    # Display Output Banners
    if prediction[0] == 1:
        st.success(f"🎉 **Decision Result: APPROVED** (Confidence: {probabilities[1] * 100:.2f}%)")
    else:
        st.error(f"❌ **Decision Result: REJECTED** (Confidence: {probabilities[0] * 100:.2f}%)")
