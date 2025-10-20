import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor

# Title
st.set_page_config(page_title="House Price Predictor", layout="wide")
st.title("🏠 House Price Prediction App")
st.markdown("Predict the price of your dream house based on features.")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv('Housing.csv')
    binary_map = {'yes': 1, 'no': 0}
    for col in ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']:
        df[col] = df[col].map(binary_map)
    df_dummies = pd.get_dummies(df['furnishingstatus'], prefix='status')
    df = pd.concat([df, df_dummies], axis=1)
    binary_map2 = {True: 1, False: 0}
    for col in ['status_furnished', 'status_semi-furnished', 'status_unfurnished']:
        df[col] = df[col].map(binary_map2)
    df.drop(columns=['furnishingstatus'], inplace=True)
    return df

df = load_data()

# Split data
X = df.drop('price', axis=1)
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# KNN model
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# ----------------- User Input -----------------
st.header("Predict Your House Price 🏡")

def user_input_features():
    col1, col2, col3 = st.columns(3)
    
    with col1:
        area = st.number_input("Area (sq ft)", min_value=100, value=1000)
        bedrooms = st.number_input("Bedrooms", min_value=1, value=3)
        bathrooms = st.number_input("Bathrooms", min_value=1, value=2)
        stories = st.number_input("Stories", min_value=1, value=1)
    
    with col2:
        mainroad = st.radio("Main Road", ["yes", "no"])
        guestroom = st.radio("Guest Room", ["yes", "no"])
        basement = st.radio("Basement", ["yes", "no"])
        hotwaterheating = st.radio("Hot Water Heating", ["yes", "no"])
    
    with col3:
        airconditioning = st.radio("Air Conditioning", ["yes", "no"])
        parking = st.number_input("Parking Spaces", min_value=0, value=1)
        prefarea = st.radio("Preferred Area", ["yes", "no"])
        furnishingstatus = st.radio("Furnishing Status", ["furnished", "semi-furnished", "unfurnished"])
    
    # Map binary and one-hot encode furnishing status
    binary_map = {'yes': 1, 'no': 0}
    features = {
        'area': area,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'stories': stories,
        'mainroad': binary_map[mainroad],
        'guestroom': binary_map[guestroom],
        'basement': binary_map[basement],
        'hotwaterheating': binary_map[hotwaterheating],
        'airconditioning': binary_map[airconditioning],
        'parking': parking,
        'prefarea': binary_map[prefarea],
        'status_furnished': 1 if furnishingstatus=='furnished' else 0,
        'status_semi-furnished': 1 if furnishingstatus=='semi-furnished' else 0,
        'status_unfurnished': 1 if furnishingstatus=='unfurnished' else 0
    }
    return pd.DataFrame([features])

input_df = user_input_features()
input_scaled = scaler.transform(input_df)
predicted_price = knn.predict(input_scaled)

st.subheader("💰 Predicted House Price:")

st.metric(label="Estimated Price", value=f"₹ {predicted_price[0]:,.0f}")
