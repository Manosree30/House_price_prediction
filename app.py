import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- Load trained model, scaler, and feature list ---
with open('rf_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('feature_columns.pkl', 'rb') as f:
    model_features = pickle.load(f)

st.set_page_config(page_title="🏠 House Price Predictor", layout="wide")
st.title("🏠 House Price Prediction App")
st.markdown("Predict the price of a house based on its features.")

# --- Sidebar Inputs ---
st.sidebar.header("Enter House Details")

col1, col2, col3 = st.sidebar.columns(3)
with col1:
    area = st.number_input("Area (sq ft)", min_value=300, max_value=20000, value=3000)
    bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
with col2:
    bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)
    stories = st.number_input("Stories", min_value=1, max_value=5, value=2)
with col3:
    parking = st.number_input("Parking", min_value=0, max_value=5, value=1)

mainroad = st.sidebar.radio("Main Road?", ["No", "Yes"]) == "Yes"
guestroom = st.sidebar.radio("Guest Room?", ["No", "Yes"]) == "Yes"
basement = st.sidebar.radio("Basement?", ["No", "Yes"]) == "Yes"
hotwaterheating = st.sidebar.radio("Hot Water Heating?", ["No", "Yes"]) == "Yes"
airconditioning = st.sidebar.radio("Air Conditioning?", ["No", "Yes"]) == "Yes"
prefarea = st.sidebar.radio("Preferred Area?", ["No", "Yes"]) == "Yes"
furnishingstatus = st.sidebar.radio("Furnishing Status", ["Unfurnished", "Semi-furnished", "Furnished"])

# --- Construct input DataFrame ---
input_dict = {
    'area': area,
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'stories': stories,
    'parking': parking,
    'mainroad_yes': int(mainroad),
    'guestroom_yes': int(guestroom),
    'basement_yes': int(basement),
    'hotwaterheating_yes': int(hotwaterheating),
    'airconditioning_yes': int(airconditioning),
    'prefarea_yes': int(prefarea),
    'furnishingstatus_semi-furnished': 1 if furnishingstatus=="Semi-furnished" else 0,
    'furnishingstatus_unfurnished': 1 if furnishingstatus=="Unfurnished" else 0
}

# --- Feature engineering ---
input_dict['area_per_story'] = input_dict['area'] / input_dict['stories'] if input_dict['stories'] else 0
input_dict['bed_bath_ratio'] = input_dict['bedrooms'] / input_dict['bathrooms'] if input_dict['bathrooms'] else 0
input_dict['rooms_per_story'] = input_dict['bedrooms'] / input_dict['stories'] if input_dict['stories'] else 0
input_dict['parking_per_story'] = input_dict['parking'] / input_dict['stories'] if input_dict['stories'] else 0

amenities_cols = ['mainroad_yes','guestroom_yes','basement_yes','hotwaterheating_yes','airconditioning_yes','prefarea_yes']
input_dict['amenities_score'] = sum([input_dict[col] for col in amenities_cols])

input_df = pd.DataFrame([input_dict])

# --- Align features ---
for col in model_features:
    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[model_features]

# --- Scale numeric features ---
numeric_cols = ['area','bedrooms','bathrooms','stories','parking',
                'area_per_story','bed_bath_ratio','rooms_per_story','parking_per_story','amenities_score']
input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

# --- Predict ---
if st.button("💰 Predict House Price"):
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted House Price: ₹{prediction:,.0f}")
    st.balloons()
