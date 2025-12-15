import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# --- 1. Load the Saved Model and Scaler ---
# Model ko Load karna jo pichle step mein save kiya gaya tha
try:
    with open("fraud_detection_voting_model.pkl", "rb") as f:
        data = pickle.load(f)
        voting_model = data["model"]
        scaler = data["scaler"]
        feature_names = data["feature_names"]
except FileNotFoundError:
    st.error("Error: Model file 'fraud_detection_voting_model.pkl' not found. Please run the model training code first.")
    st.stop()

# --- 2. Streamlit UI Setup ---
st.set_page_config(page_title="Credit Card Fraud Detection App", layout="wide")

st.title("💳 Credit Card Fraud Detection System")
st.markdown("---")

st.markdown("""
    **Is Model ka Maqsad (Goal):** Diye gaye transaction parameters ki bunyad par yeh predict karna ke transaction **Fraud (1)** hai ya **Valid (0)**.
""")

# --- 3. Input Form (User Input) ---
st.header("Transaction Parameters Input")

# Since the Fraud dataset (creditcard.csv) uses PCA features (V1-V28), 
# we need to collect all 30 features (Time, V1-V28, Amount)

# We will arrange inputs in columns for better UI layout

# Initialize a dictionary to store inputs
input_data = {}

# Time and Amount (These features are highly non-linear, but must be collected)
col_time, col_amount = st.columns(2)
with col_time:
    input_data['Time'] = st.number_input(
        "Time (Seconds elapsed since first transaction):",
        min_value=0.0, max_value=200000.0, value=1000.0, step=100.0, format="%.2f"
    )
with col_amount:
    input_data['Amount'] = st.number_input(
        "Transaction Amount ($):",
        min_value=0.0, max_value=26000.0, value=100.0, step=10.0, format="%.2f"
    )

# V1 to V28 (These are anonymized features from PCA)
# Grouping them into 4 columns for clean UI
st.subheader("PCA Features (V1 - V28)")

cols_v = st.columns(4)

for i in range(1, 29):
    feature = f'V{i}'
    # Determine which column to place the input in
    col_index = (i - 1) % 4
    
    with cols_v[col_index]:
        # Using a wide range for V features since their actual range is not clear in a simple way
        # Default value set to 0.0, min/max based on typical dataset ranges
        input_data[feature] = st.number_input(
            f"{feature}:",
            min_value=-50.0, max_value=50.0, value=0.0, step=0.1, format="%.6f"
        )

# --- 4. Prediction Button ---
st.markdown("---")
if st.button("Predict Transaction Status", help="Click to predict if the transaction is Fraud or Valid"):
    
    # Convert input data to DataFrame in the correct feature order
    # This is crucial: the order of columns must match the training data
    input_df = pd.DataFrame([input_data])
    input_df = input_df[feature_names] 
    
    # --- Scaling (Only Time and Amount) ---
    # Apply the same scaling ONLY to 'Time' and 'Amount' features as done in training.
    # The 'V' columns are already scaled in the dataset.
    
    # Make a copy for scaling
    scaled_input = input_df.copy()
    
    # Prepare the data for scaling (only 'Time' and 'Amount')
    data_to_scale = scaled_input[['Time', 'Amount']]
    
    # Scale and replace the values in the input
    scaled_values = scaler.transform(data_to_scale)
    scaled_input['Time'] = scaled_values[:, 0]
    scaled_input['Amount'] = scaled_values[:, 1]
    
    # --- Prediction ---
    prediction = voting_model.predict(scaled_input.values)
    prediction_proba = voting_model.predict_proba(scaled_input.values)

    st.subheader("Prediction Result")
    
    # Display Result based on prediction (0 or 1)
    if prediction[0] == 1:
        st.error(f"❌ Warning: This Transaction is Predicted as **FRAUD** (Risk Score: {prediction_proba[0][1]*100:.2f}%)")
        st.markdown("<p style='font-size: 18px; color: red;'>**Action Required:** Transaction Blocked and flagged for review.</p>", unsafe_allow_html=True)
    else:
        st.success(f"✅ Status: This Transaction is Predicted as **VALID** (Risk Score: {prediction_proba[0][1]*100:.2f}%)")
        st.markdown("<p style='font-size: 18px; color: green;'>**Action:** Transaction Approved.</p>", unsafe_allow_html=True)

    # Optional: Display probability for confidence
    st.markdown("---")
    st.info(f"Model Confidence: Valid (0) = {prediction_proba[0][0]*100:.2f}% | Fraud (1) = {prediction_proba[0][1]*100:.2f}%")