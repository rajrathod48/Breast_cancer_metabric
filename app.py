import streamlit as st
import pandas as pd
import numpy as np
from joblib import load

# Load the trained model
model = load("best_rf.joblib")

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
    'ER Status': [1 if er_status == 'Positive' else 0],
    'PR Status': [1 if pr_status == 'Positive' else 0],
    'HER2 Status': [1 if her2_status == 'Positive' else 0]
})

# Make prediction when button is clicked
if st.button("Predict Survival Status"):
    # Scale the data (scaling might have been applied earlier)
    # Assuming you used StandardScaler during training
    scaled_data = data  # Replace this with actual scaling if needed
    
    # Make prediction
    prediction = model.predict(scaled_data)
    
    # Output the result
    if prediction == 1:
        st.success("The patient is predicted to survive.")
    else:
        st.error("The patient is predicted not to survive.")
