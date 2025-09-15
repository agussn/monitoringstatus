import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

# ==== Load Dataset & Train Model (sekali jalan) ====
@st.cache_resource
def load_model():
    df = pd.read_csv("dataset_dokumen_100.csv")
    X = df[['dok_dalam', 'dok_luar', 'dok_campuran']]
    y = df['status']

    model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    model.fit(X, y)
    return model

model = load_model()

# ==== UI Streamlit ====
st.title("Monitoring Status Dokumen Akreditasi")
st.write("Masukkan persentase dokumen yang **belum tersedia**:")

dok_dalam = st.slider("Dokumen Dalam (%)", 0, 100, 10)
dok_luar = st.slider("Dokumen Luar (%)", 0, 100, 10)
dok_campuran = st.slider("Dokumen Campuran (%)", 0, 100, 10)

if st.button("Prediksi Status"):
    pred = model.predict([[dok_dalam, dok_luar, dok_campuran]])[0]
    st.success(f"Status prediksi: **{pred}**")
