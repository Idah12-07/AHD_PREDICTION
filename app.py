import streamlit as st
import numpy as np
import joblib

# ==========================
# Load trained hybrid model
# ==========================
model = joblib.load("ahd_model_C_hybrid.pkl")

# ==========================
# Streamlit Page Config
# ==========================
st.set_page_config(
    page_title="AHD Detection",
    layout="wide",
    page_icon="🧠"
)

# Sidebar - Input section
st.sidebar.header("📝 Patient Information")

# Manual input fields
age = st.sidebar.number_input("Age at Reporting", min_value=0, max_value=100, value=35)
weight = st.sidebar.number_input("Weight (kg)", min_value=20.0, max_value=150.0, value=60.0)
height = st.sidebar.number_input("Height (cm)", min_value=100, max_value=220, value=165)
cd4 = st.sidebar.number_input("Latest CD4 Count", min_value=0, max_value=2000, value=350)
vl = st.sidebar.number_input("Latest Viral Load (copies/ml)", min_value=0, max_value=500000, value=1000)
who_stage = st.sidebar.selectbox("Last WHO Stage", [1, 2, 3, 4])
months_rx = st.sidebar.slider("Months of Prescription", 1, 12, 3)
cd4_risk = st.sidebar.selectbox("CD4 Risk Category", ["Severe", "Moderate", "Normal"])

# ==========================
# Feature engineering
# ==========================
cd4_risk_severe = 1 if cd4_risk == "Severe" else 0
cd4_risk_moderate = 1 if cd4_risk == "Moderate" else 0
cd4_risk_normal = 1 if cd4_risk == "Normal" else 0

bmi = weight / ((height / 100) ** 2)
vl_suppressed = 1 if vl < 1000 else 0
cd4_missing = 0 if cd4 > 0 else 1
vl_missing = 0 if vl > 0 else 1

# Feature vector (must match model input)
input_data = np.array([[
    age, weight, height, bmi, cd4, cd4_missing, vl, vl_suppressed,
    vl_missing, who_stage, months_rx,
    cd4_risk_moderate, cd4_risk_normal, cd4_risk_severe
]])

# ==========================
# Main Page Layout
# ==========================
st.title("🧠 Advanced HIV Disease (AHD) Detection")
st.markdown("This tool helps clinicians assess the **risk of Advanced HIV Disease (AHD)** based on patient details.")

# Info dropdown
with st.expander("ℹ️ About the Prediction"):
    st.markdown("""
    - **Yes** = Patient is at risk of Advanced HIV Disease (AHD).  
    - **No** = Patient is not at risk.  
    - **Probability** = Confidence of the model in predicting AHD risk.  
    - **Risk Levels**  
        - 🟢 Low Risk: Safe to continue routine care  
        - 🟠 Moderate Risk: Requires closer monitoring  
        - ⚠️ High Risk: Consider immediate clinical review  

    ⚡ *Note: This tool is a decision-support system. Always use clinical judgment alongside the results.*
    """)

# Patient summary
with st.expander("📋 Patient Summary", expanded=True):
    st.write(f"- **Age:** {age} years")
    st.write(f"- **Weight:** {weight} kg")
    st.write(f"- **Height:** {height} cm (BMI: {bmi:.1f})")
    st.write(f"- **Latest CD4 Count:** {cd4}")
    st.write(f"- **Latest Viral Load:** {vl} copies/ml ({'Suppressed' if vl_suppressed else 'Not Suppressed'})")
    st.write(f"- **WHO Stage:** {who_stage}")
    st.write(f"- **Months of Prescription:** {months_rx}")
    st.write(f"- **CD4 Risk Category:** {cd4_risk}")

# Prediction button
if st.sidebar.button("🔍 Predict AHD Risk"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("📌 Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("AHD Risk", "Yes" if prediction == 1 else "No")
        st.metric("Risk Probability", f"{probability:.2%}")

    with col2:
        st.progress(probability)
        if probability > 0.7:
            st.error("⚠️ High Risk – Consider immediate clinical review.")
        elif probability > 0.4:
            st.warning("🟠 Moderate Risk – Monitor closely.")
        else:
            st.success("🟢 Low Risk – Continue routine care.")

# ==========================
# Footer (Watermark / Credit)
# ==========================
st.markdown(
    """
    <hr style="border:1px solid #ccc;">
    <div style="text-align:center; font-size:13px; color:gray;">
    Built with ❤️ by <b>Idah Anyango</b>
    </div>
    """,
    unsafe_allow_html=True
)
