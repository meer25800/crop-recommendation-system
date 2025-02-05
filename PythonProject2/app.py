import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import requests
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from time import sleep
import sys
import subprocess

# Ensure Plotly is installed
try:
    import plotly.express as px
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "plotly"])
    import plotly.express as px

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
            background: linear-gradient(to right, #2B3A3A, #3E4C4C);
            color: #ffffff;
        }
        .big-font {
            font-size: 28px !important;
            font-weight: bold;
            color: #2BCEEE;
        }
        .sidebar .sidebar-content {
            background-color: #1F2D2D;
            border-radius: 10px;
            padding: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# --- Title ---
st.markdown("<h1 class='big-font'>🌾 Crop Recommendation System</h1>", unsafe_allow_html=True)
st.markdown("🔍 Select soil & climate parameters to get the best crop recommendations.")

# --- Load Dataset ---
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/meer25800/crop-recommendation-system/main/PythonProject2/dataset/Crop_recommendation.csv"
    return pd.read_csv(url)

data = load_data()
label_encoder = LabelEncoder()
data['label'] = label_encoder.fit_transform(data['label'])

# --- Train Multiple Models ---
@st.cache_resource
def train_models():
    features = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    X = data[features]
    y = data["label"]
    
    models = {
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "SVM": SVC(probability=True),
        "KNN": KNeighborsClassifier(),
        "Logistic Regression": LogisticRegression(),
        "Naive Bayes": GaussianNB()
    }
    
    for model in models.values():
        model.fit(X, y)
    
    return models

models = train_models()

# --- Sidebar for Inputs ---
st.sidebar.header("🌿 Select Soil & Climate Conditions")

N = st.sidebar.slider("Nitrogen (N)", 0, 140, 50)
P = st.sidebar.slider("Phosphorus (P)", 5, 130, 50)
K = st.sidebar.slider("Potassium (K)", 5, 85, 50)
temperature = st.sidebar.slider("Temperature (°C)", 8.0, 45.0, 25.0)
humidity = st.sidebar.slider("Humidity (%)", 10.0, 100.0, 50.0)
ph = st.sidebar.slider("pH Level", 2.0, 10.0, 7.0)
rainfall = st.sidebar.slider("Rainfall (mm)", 20.0, 220.0, 100.0)

model_choice = st.sidebar.selectbox("🧠 Choose Model", ["Decision Tree", "Random Forest", "SVM","KNN","Logistic Regression", "Naive Bayes"])

# --- Prediction ---
if st.sidebar.button("🌱 Recommend Crop"):
    st.sidebar.info("Processing... Please wait.")
    sleep(2)  # Simulate loading time
    user_input = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    selected_model = models[model_choice]
    probs = selected_model.predict_proba(user_input)[0]
    top_3_indices = np.argsort(probs)[-3:][::-1]
    top_3_crops = label_encoder.inverse_transform(top_3_indices)
    
    st.sidebar.success(f"✅ Best Crop: **{top_3_crops[0]}**")
    st.sidebar.write(f"🥈 Second Best: {top_3_crops[1]}")
    st.sidebar.write(f"🥉 Third Best: {top_3_crops[2]}")
    
    # --- Radar Chart ---
    params = [N, P, K, temperature, humidity, ph, rainfall]
    param_labels = ["Nitrogen", "Phosphorus", "Potassium", "Temp", "Humidity", "pH", "Rainfall"]
    fig = px.line_polar(r=dict(zip(param_labels, params)), theta=param_labels, line_close=True)
    fig.update_traces(fill='toself', line=dict(color='blue'))
    fig.update_layout(title=f"📊 Soil & Climate Parameters for {top_3_crops[0]}")
    st.plotly_chart(fig)

    # --- Gauge Meter for Confidence ---
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=max(probs) * 100,
        title={'text': "Confidence Level (%)"},
        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "green"}}
    ))
    st.plotly_chart(fig_gauge)

# --- Show Dataset ---
st.markdown("<h2 class='big-font'>📊 Crop Dataset</h2>", unsafe_allow_html=True)
if st.checkbox("Show Dataset"):
    st.dataframe(data.drop(columns=['label']).style.set_properties(**{'background-color': '#DFFFD6', 'color': 'black'}))
