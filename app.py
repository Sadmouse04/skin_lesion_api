import streamlit as st
import requests
import os
from PIL import Image

N8N = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Skin Lesion Triage", layout="wide")
st.title("Skin Lesion Predictor")
st.markdown("Enter patient data below.")

# Sidebar for Inputs
st.sidebar.header("Patient Data")

image_dir = "data/images/"

# Image ID (Must exist in your dataset folder)
img_id = st.sidebar.text_input("Image Filename", "PAT_56_86_479.png")

# Age
age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=45)

# Categorical toggles
all_regions = ["ARM", "NECK", "FACE", "HAND", "FOREARM", "CHEST", "NOSE", 
    "THIGH", "SCALP", "EAR", "BACK", "FOOT", "ABDOMEN", "LIP"]
region = st.sidebar.selectbox("Region", all_regions)

itch = st.sidebar.radio("Itchy?", ["Yes", "No"])
grew = st.sidebar.radio("Lesion Grew?", ["Yes", "No"])
hurt = st.sidebar.radio("Hurts / Painful?", ["Yes", "No"])
changed = st.sidebar.radio("Changed Shape / Color?", ["Yes", "No"])
bleed = st.sidebar.radio("Bleeding?", ["Yes", "No"])
elevation = st.sidebar.radio("Elevated?", ["Yes", "No"])

if st.button("Run Prediction & Triage"):
    # Construct the payload
    payload = {
        "img_id": f"data/images/{img_id}",
        "age": float(age),
        "region_ARM": 1 if region == "ARM" else 0,
        "region_NECK": 1 if region == "NECK" else 0,
        "region_FACE": 1 if region == "FACE" else 0,
        "region_HAND": 1 if region == "HAND" else 0,
        "region_FOREARM": 1 if region == "FOREARM" else 0,
        "region_CHEST": 1 if region == "CHEST" else 0,
        "region_NOSE": 1 if region == "NOSE" else 0,
        "region_THIGH": 1 if region == "THIGH" else 0,
        "region_SCALP": 1 if region == "SCALP" else 0,
        "region_EAR": 1 if region == "EAR" else 0,
        "region_BACK": 1 if region == "BACK" else 0,
        "region_FOOT": 1 if region == "FOOT" else 0,
        "region_ABDOMEN": 1 if region == "ABDOMEN" else 0,
        "region_LIP": 1 if region == "LIP" else 0,

        "itch_False": 1 if itch == "No" else 0,
        "itch_True": 1 if itch == "Yes" else 0,
        "grew_False": 1 if grew == "No" else 0,
        "grew_True": 1 if grew == "Yes" else 0,
        "hurt_False": 1 if hurt == "No" else 0,
        "hurt_True": 1 if hurt == "Yes" else 0,
        "changed_False": 1 if changed == "No" else 0,
        "changed_True": 1 if changed == "Yes" else 0,
        "bleed_False": 1 if bleed == "No" else 0,
        "bleed_True": 1 if bleed == "Yes" else 0,
        "elevation_False": 1 if elevation == "No" else 0,
        "elevation_True": 1 if elevation == "Yes" else 0
    }

    # Create two columns for the UI
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Lesion Image")
        img_path = os.path.join(image_dir, img_id)
        if os.path.exists(img_path):
            image = Image.open(img_path)
            st.image(image, caption=img_id, use_container_width=True)
        else:
            st.error(f"Image not found at: {img_path}")
            st.info("Make sure the image is located in the dataset folder!")

    with col2:
        with st.spinner("Analyzing patient data..."):
            response = requests.post(N8N, json=payload, timeout=120)

            result = response.json()
                    
            st.success("Analysis Complete!")
                    
            # Display Binary Risk Score
            risk_color = "red" if result["binary_prediction"] == "Malignant" else "green"
            st.markdown(f"### Risk Level: :{risk_color}[**{result['binary_prediction']}**]")
            st.progress(result["binary_confidence"], text=f"Confidence: {result['binary_confidence']*100:.1f}%")
                    
            st.divider()

            # Display Multi-class Details
            st.subheader("Detailed Pathology")
            metric_col1, metric_col2 = st.columns(2)
            metric_col1.metric("Specific Diagnosis", result["multi_class_prediction"])
            metric_col2.metric("Class Confidence", f"{result['multi_class_confidence']*100:.1f}%")
                    
            st.bar_chart(result["raw_probabilities"])

            st.divider()

            # AI Agent's recommendations
            st.subheader("AI Agent Recommendations")
            st.write(f"**Recommended Action:** {result['recommended_action']}")
            st.text_area("Draft Triage Note (Ready for Doctor Review):", result['draft_message'], height=150)
                    