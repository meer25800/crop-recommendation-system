import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("dataset/Crop_recommendation.csv")

data = load_data()

# Train the model
@st.cache_resource
def train_model():
    features = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    X = data[features]
    y = data["label"]
    model = DecisionTreeClassifier()
    model.fit(X, y)
    return model

model = train_model()

# Streamlit UI
st.title("🌱 Crop Recommendation System")

st.sidebar.header("Enter Soil & Climate Conditions 🌿")

# Sidebar inputs
N = st.sidebar.slider("Nitrogen (N)", 0, 100, 50)
P = st.sidebar.slider("Phosphorus (P)", 0, 100, 50)
K = st.sidebar.slider("Potassium (K)", 0, 100, 50)
temperature = st.sidebar.slider("Temperature (°C)", 0.0, 50.0, 25.0)
humidity = st.sidebar.slider("Humidity (%)", 0.0, 100.0, 50.0)
ph = st.sidebar.slider("pH Level", 0.0, 14.0, 7.0)
rainfall = st.sidebar.slider("Rainfall (mm)", 0.0, 500.0, 100.0)

# Predict crop automatically
user_input = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
prediction = model.predict(user_input)
st.success(f"🌾 Recommended Crop: **{prediction[0]}**")

# Dataset exploration
st.subheader("📊 Dataset Exploration")
if st.checkbox("Show Dataset"):
    st.dataframe(data)

# Filtering by Crop
crop_list = data["label"].unique()
selected_crop = st.selectbox("Filter Dataset by Crop:", ["All"] + list(crop_list))
if selected_crop != "All":
    st.dataframe(data[data["label"] == selected_crop])

# Visualization
st.subheader("📈 Data Insights")
if st.checkbox("Show Correlation Heatmap"):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(data.drop(columns=["label"]).corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
