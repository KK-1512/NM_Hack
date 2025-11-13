import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
import io
import requests
from PIL import Image
from io import BytesIO

st.set_page_config(page_title="AI Material Selector", layout="centered")

# ---- Title and Intro ----
st.title("AI-Based Material Selection for Bottle Holder")
st.write("""
This app helps designers choose the **best material** for a universal bottle holder 
by balancing strength, weight, cost, and sustainability using a data-driven scoring system.  
**Done By:** Hasitha S, Kowshic K T, Krishnakumar V
""")

# ---- Display Project Image ----
IMAGE_URL = "https://raw.githubusercontent.com/KK-1512/NM_Hack/main/bottle_holder.jpg"  # 🔁 Replace with your actual image name

try:
    response = requests.get(IMAGE_URL, timeout=10)
    img = Image.open(BytesIO(response.content))
    st.image(img, caption="Universal Bottle Holder Prototype", use_container_width=True)
except Exception as e:
    st.warning("⚠️ Could not load image. Please check the URL or file name in GitHub.")
    st.write(e)

# ---- Load Dataset ----
DATA_URL = "https://raw.githubusercontent.com/KK-1512/NM_Hack/main/material_selection_dataset.csv"

@st.cache_data
def load_data():
    try:
        df = pd.read_csv(DATA_URL)
        return df
    except Exception as e:
        st.error(f"Could not load CSV from GitHub: {e}")
        return None

df = load_data()

# Option to upload your own dataset
uploaded = st.file_uploader("Upload your own materials CSV (optional)", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)

if df is not None:
    st.subheader("Material Dataset")
    st.dataframe(df)

    # ---- Weight sliders ----
    st.subheader("Weight Configuration")
    w_strength_to_weight = st.slider("Weight for Strength-to-Weight Ratio", 0.0, 1.0, 0.6)
    w_sustainability = 1.0 - w_strength_to_weight
    st.write(f"Weight distribution → Strength-to-weight: {w_strength_to_weight:.2f}, Sustainability: {w_sustainability:.2f}")

    # ---- Compute Scores ----
    df = df.copy()
    df['Strength_to_Weight'] = df['Tensile Strength (MPa)'] / df['Density (g/cc)']
    df['norm_stw'] = (df['Strength_to_Weight'] - df['Strength_to_Weight'].min()) / (df['Strength_to_Weight'].max() - df['Strength_to_Weight'].min() + 1e-9)
    df['norm_sust'] = (df['Sustainability (1–10)'] - df['Sustainability (1–10)'].min()) / (df['Sustainability (1–10)'].max() - df['Sustainability (1–10)'].min() + 1e-9)
    df['Performance_Score'] = w_strength_to_weight * df['norm_stw'] + w_sustainability * df['norm_sust']

    # ---- Show Ranking ----
    st.subheader("Top Recommended Materials")
    top_n = st.slider("Show Top N Materials", 1, min(15, len(df)), 5)
    ranked = df.sort_values(by='Performance_Score', ascending=False).reset_index(drop=True)
    st.table(ranked[['Material', 'Density (g/cc)', 'Tensile Strength (MPa)', 'Sustainability (1–10)', 'Performance_Score']].head(top_n))

    # ---- Optional: ML Prediction ----
    st.subheader("Predict Material Score using ML Model")
    if st.button("Train Quick Model"):
        X = df[['Density (g/cc)', 'Tensile Strength (MPa)', 'Elastic Modulus (GPa)', 'Cost (₹/kg)', 'Sustainability (1–10)']]
        y = df['Performance_Score']
        model = RandomForestRegressor(random_state=42)
        model.fit(X, y)
        joblib.dump(model, "material_selector.pkl")
        st.success("Model trained successfully and saved as material_selector.pkl!")

    # ---- Material Selection ----
    selected = st.selectbox("Select a Material to View Details", ranked['Material'])
    st.write(ranked[ranked['Material'] == selected].T)

else:
    st.warning("Please upload a CSV file to continue.")
