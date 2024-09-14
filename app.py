import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
from sklearn.preprocessing import StandardScaler

# Load the trained model and scaler
model = load("best_rf.joblib")
scaler = load("scaler.joblib")

# Title of the app
st.title("Breast Cancer Survival Prediction (METABRIC)")

# Input form for user to provide patient data
st.header("Enter Patient Information")

# Assuming these are some of the relevant features; adjust according to your dataset
age = st.number_input("Age", min_value=18, max_value=100, value=50)
tumor_size = st.number_input("Tumor Size (in mm)", min_value=0, max_value=200, value=20)
lymph_nodes = st.number_input("Positive Lymph Nodes", min_value=0, max_value=50, value=2)
er_status = st.selectbox("ER Status", ['Positive', 'Negative'])
pr_status = st.selectbox("PR Status", ['Positive', 'Negative'])
her2_status = st.selectbox("HER2 Status", ['Positive', 'Negative'])

# Map inputs into a DataFrame (adjust feature names to match your dataset)
data = pd.DataFrame({
    'Age': [age],
    'Tumor Size': [tumor_size],
    'Positive Lymph Nodes': [lymph_nodes],
    'ER Status_Positive': [1 if er_status == 'Positive' else 0],
    'PR Status_Positive': [1 if pr_status == 'Positive' else 0],
    'HER2 Status_Positive': [1 if her2_status == 'Positive' else 0]
})

# Align columns to match the training data structure (make sure the columns match exactly)
columns = ['Age', 'Tumor Size', 'Positive Lymph Nodes', 'ER Status_Positive', 'PR Status_Positive', 'HER2 Status_Positive']
data = data.reindex(columns=columns, fill_value=0)

# Scale the input data
scaled_data = scaler.transform(data)

# Make prediction when button is clicked
if st.button("Predict Survival Status"):
    # Make prediction
    prediction = model.predict(scaled_data)
    
    # Output the result
    if prediction == 1:
        st.success("The patient is predicted to survive.")
    else:
        st.error("The patient is predicted not to survive.")
