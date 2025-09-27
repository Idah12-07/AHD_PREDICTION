import streamlit as st
import numpy as np
import joblib

# Load Model C
model = joblib.load("ahd_model_C.pkl")

st.set_page_config(page_title="AHD DETECTION", layout="centered")

st.title("🧠 AHD DETECTION")
st.markdown("Enter patient details below to assess risk of Advanced HIV Disease (AHD).")

# Manual input fields
age = st.number_input("Age at Reporting", min_value=0, max_value=100, value=35)
weight = st.number_input("Weight (kg)", min_value=20.0, max_value=150.0, value=60.0)
height = st.number_input("Height (cm)", min_value=100, max_value=220, value=165)
cd4 = st.number_input("Latest CD4 Count", min_value=0, max_value=2000, value=350)
vl = st.number_input("Latest Viral Load (copies/ml)", min_value=0, max_value=500000, value=1000)
who_stage = st.selectbox("Last WHO Stage", [1, 2, 3, 4])
months_rx = st.slider("Months of Prescription", 1, 12, 3)

# Derived features
bmi = weight / ((height / 100) ** 2)
vl_suppressed = 1 if vl < 1000 else 0
cd4_missing = 0 if cd4 > 0 else 1
vl_missing = 0 if vl > 0 else 1

# Feature vector (must match model input)
input_data = np.array([[age, weight, height, bmi, cd4, cd4_missing, vl, vl_suppressed,
                        vl_missing, who_stage, months_rx]])

# Predict
if st.button("Predict AHD Risk"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("📌 Prediction Result")
    st.write(f"**AHD Risk:** {'Yes' if prediction == 1 else 'No'}")
    st.write(f"**Risk Probability:** {probability:.2f}")

    # Visual feedback
    st.progress(probability)
    if probability > 0.7:
        st.error("⚠️ High Risk – Consider immediate clinical review.")
    elif probability > 0.4:
        st.warning("🟠 Moderate Risk – Monitor closely.")
    else:
        st.success("🟢 Low Risk – Continue routine care.")

