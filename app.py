import streamlit as st
import pandas as pd
import pickle
import numpy as np

st.set_page_config(page_title="House Price Predictor", layout="wide")
st.title("🏠 House Price Prediction (k-NN Regressor)")

# Load saved model and scaler
with open('knn_model.pkl', 'rb') as f:
    knn_loaded = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler_loaded = pickle.load(f)

binary_map = {"Yes": 1, "No": 0}
furnish_map = {"Unfurnished": 0, "Furnished": 1}

# Layout
col1, col2, col3 = st.columns(3)

with col1:
    area = st.slider("Area (1000s sq.ft)", 7.0, 10.0, 8.5, 0.01)
    bedrooms = st.slider("Bedrooms", 1, 5, 3)
    bathrooms = st.slider("Bathrooms", 1, 4, 2)
    stories = st.slider("Stories", 1, 4, 2)

with col2:
    mainroad = st.selectbox("Main Road", ["Yes", "No"])
    guestroom = st.selectbox("Guest Room", ["Yes", "No"])
    basement = st.selectbox("Basement", ["Yes", "No"])
    hotwaterheating = st.selectbox("Hot Water Heating", ["Yes", "No"])

with col3:
    airconditioning = st.selectbox("Air Conditioning", ["Yes", "No"])
    parking = st.slider("Parking", 0, 3, 1)
    prefarea = st.selectbox("Preferred Area", ["Yes", "No"])
    furnishingstatus = st.selectbox("Furnishing Status", ["Unfurnished", "Furnished"])

# Prepare input DataFrame
input_df = pd.DataFrame({
    'area': [area],
    'bedrooms': [bedrooms],
    'bathrooms': [bathrooms],
    'stories': [stories],
    'mainroad': [binary_map[mainroad]],
    'guestroom': [binary_map[guestroom]],
    'basement': [binary_map[basement]],
    'hotwaterheating': [binary_map[hotwaterheating]],
    'airconditioning': [binary_map[airconditioning]],
    'parking': [parking],
    'prefarea': [binary_map[prefarea]],
    'furnishingstatus': [furnish_map[furnishingstatus]]
})

numeric_cols = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']

# Apply same log1p transform used in training
input_df[numeric_cols] = np.log1p(input_df[numeric_cols])

# Predict button
if st.button("🔍 Predict Price"):
    input_scaled = scaler_loaded.transform(input_df)
    predicted_price = knn_loaded.predict(input_scaled)[0]

    # Format price
    if predicted_price >= 1e7:
        formatted_price = f"₹{predicted_price / 1e7:.2f} Cr"
    elif predicted_price >= 1e5:
        formatted_price = f"₹{predicted_price / 1e5:.2f} Lakh"
    else:
        formatted_price = f"₹{predicted_price:,.0f}"

    st.markdown("---")
    st.markdown(f"## 💰 Predicted House Price: {formatted_price}")
