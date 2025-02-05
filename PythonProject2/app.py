import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.tree import DecisionTreeClassifier

st.title("Crop Recommendation System")

# File uploader for dataset
uploaded_file = st.file_uploader("Upload Crop Recommendation Dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded CSV file
    data = pd.read_csv(uploaded_file)

    # Train a simple model
    @st.cache_resource
    def train_model():
        features = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
        X = data[features]
        y = data["label"]
        model = DecisionTreeClassifier()
        model.fit(X, y)
        return model

    model = train_model()

    st.write("Enter the soil and climate conditions to get a recommended crop.")

    # User inputs
    N = st.number_input("Nitrogen (N)", min_value=0, max_value=100, value=50)
    P = st.number_input("Phosphorus (P)", min_value=0, max_value=100, value=50)
    K = st.number_input("Potassium (K)", min_value=0, max_value=100, value=50)
    temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0)
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0)
    ph = st.number_input("pH Level", min_value=0.0, max_value=14.0, value=7.0)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=100.0)

    # Predict crop
    if st.button("Recommend Crop"):
        user_input = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        prediction = model.predict(user_input)
        st.success(f"Recommended Crop: {prediction[0]}")

    # Show dataset if user wants
    if st.checkbox("Show Dataset"):
        st.dataframe(data)
