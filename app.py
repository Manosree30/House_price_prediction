import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained model and scaler from files
with open('knn_model.pkl', 'rb') as f:
    knn_model = pickle.load(f)

with open('scaler_X.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.title("House Price Prediction App (k-NN)")

# Sidebar inputs for features
st.sidebar.header("Input House Features")

def user_input_features():
    area = st.sidebar.number_input("Area (sq ft)", min_value=500, max_value=20000, value=4000)
    bedrooms = st.sidebar.slider("Bedrooms", 1, 10, 3)
    bathrooms = st.sidebar.slider("Bathrooms", 1, 5, 2)
    stories = st.sidebar.slider("Stories", 1, 4, 2)
    parking = st.sidebar.slider("Parking", 0, 4, 1)
    mainroad = st.sidebar.selectbox("Main Road?", [0, 1])
    guestroom = st.sidebar.selectbox("Guest Room?", [0, 1])
    basement = st.sidebar.selectbox("Basement?", [0, 1])
    hotwaterheating = st.sidebar.selectbox("Hot Water Heating?", [0, 1])
    airconditioning = st.sidebar.selectbox("Air Conditioning?", [0, 1])
    prefarea = st.sidebar.selectbox("Preferred Area?", [0, 1])

    # Arrange features as model expects
    features = np.array([[area, bedrooms, bathrooms, stories, parking,
                          mainroad, guestroom, basement, hotwaterheating,
                          airconditioning, prefarea]])
    return features

input_data = user_input_features()

# Scale input features before prediction
input_scaled = scaler.transform(input_data)

# Predict price
predicted_price = knn_model.predict(input_scaled)[0]

st.subheader("Predicted House Price")
st.success(f"₹ {predicted_price:,.0f}")

# Load full data to show plot of predicted vs actual prices on test set
df = pd.read_csv("Housing.csv")
bin_cols = ['mainroad','guestroom','basement','hotwaterheating','airconditioning','prefarea']
for col in bin_cols:
    df[col] = df[col].map({'yes':1,'no':0})

num_cols = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking'] + bin_cols
X = df[num_cols]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_test_scaled = scaler.transform(X_test)
y_pred_test = knn_model.predict(X_test_scaled)

st.subheader("Predicted vs Actual Prices (Test Set)")
fig, ax = plt.subplots(figsize=(7,5))
sns.scatterplot(x=y_test, y=y_pred_test, ax=ax)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
ax.set_xlabel("Actual Price")
ax.set_ylabel("Predicted Price")
ax.set_title("Predicted vs Actual House Prices")
st.pyplot(fig)
