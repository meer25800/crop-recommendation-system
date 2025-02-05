import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

# --- Page Config ---
st.set_page_config(
    page_title="🌾 Crop Recommendation System",
    page_icon="🌱",
    layout="wide"
)

# --- Custom Styles ---
st.markdown("""
    <style>
        .stApp {
            background-color: #f4f4f4;
        }
        .big-font {
            font-size: 24px !important;
            font-weight: bold;
            color: #2E8B57;
        }
        .small-font {
            font-size: 18px !important;
            font-weight: normal;
        }
    </style>
""", unsafe_allow_html=True)

# --- Title ---
st.markdown("<h1 class='big-font'>🌾 Crop Recommendation System</h1>", unsafe_allow_html=True)
st.write("🔍 Enter the soil and climate conditions to get a recommended crop.")

# --- Load Dataset ---
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/meer25800/crop-recommendation-system/main/PythonProject2/dataset/Crop_recommendation.csv"
    return pd.read_csv(url)

data = load_data()

# --- Train Model ---
@st.cache_resource
def train_model():
    features = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    X = data[features]
    y = data["label"]
    model = DecisionTreeClassifier()
    model.fit(X, y)
    return model

model = train_model()

# --- Sidebar for Inputs ---
st.sidebar.header("📝 Input Parameters")

N = st.sidebar.slider("Nitrogen (N)", min_value=0, max_value=100, value=50)
P = st.sidebar.slider("Phosphorus (P)", min_value=0, max_value=100, value=50)
K = st.sidebar.slider("Potassium (K)", min_value=0, max_value=100, value=50)
temperature = st.sidebar.slider("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0)
humidity = st.sidebar.slider("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0)
ph = st.sidebar.slider("pH Level", min_value=0.0, max_value=14.0, value=7.0)
rainfall = st.sidebar.slider("Rainfall (mm)", min_value=0.0, max_value=500.0, value=100.0)

# --- Prediction ---
if st.sidebar.button("🌱 Recommend Crop"):
    user_input = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = model.predict(user_input)
    
    st.sidebar.success(f"✅ Recommended Crop: **{prediction[0]}**")

# --- Show Dataset ---
st.markdown("<h2 class='small-font'>📊 Crop Data</h2>", unsafe_allow_html=True)
if st.checkbox("Show Dataset"):
    st.dataframe(data.style.set_properties(**{'background-color': '#f9f9f9', 'color': 'black'}))
