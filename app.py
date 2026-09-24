import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from pathlib import Path

st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Segmentation")
st.caption("K-Means Clustering pada UCI Online Retail")

DATA_FILE = Path("customer_segments.csv")
MODEL_FILE = Path("kmeans_model.pkl")
SCALER_FILE = Path("scaler.pkl")
SUMMARY_FILE = Path("cluster_summary.csv")

required = [DATA_FILE, MODEL_FILE, SCALER_FILE]
missing = [str(x) for x in required if not x.exists()]

if missing:
    st.error("File deployment belum lengkap: " + ", ".join(missing))
    st.info("Jalankan seluruh notebook terlebih dahulu agar file hasil clustering terbentuk.")
    st.stop()

df = pd.read_csv(DATA_FILE)
model = joblib.load(MODEL_FILE)
scaler = joblib.load(SCALER_FILE)

feature_cols = [
    "Recency", "Frequency", "Monetary",
    "TotalQuantity", "UniqueProducts",
    "AvgTransactionValue"
]

st.sidebar.header("Filter")
cluster_options = sorted(df["Cluster"].unique().tolist())
selected = st.sidebar.multiselect(
    "Pilih Cluster",
    cluster_options,
    default=cluster_options
)

filtered = df[df["Cluster"].isin(selected)].copy()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Jumlah Pelanggan", f"{len(filtered):,}")
c2.metric("Jumlah Cluster", f"{df['Cluster'].nunique()}")
c3.metric("Total Revenue", f"£{filtered['Monetary'].sum():,.2f}")
c4.metric("Rata-rata Revenue", f"£{filtered['Monetary'].mean():,.2f}")

st.divider()

st.subheader("Distribusi Pelanggan per Cluster")
cluster_counts = (
    filtered["Cluster"]
    .value_counts()
    .sort_index()
    .reset_index()
)
cluster_counts.columns = ["Cluster", "JumlahPelanggan"]

fig1 = px.bar(
    cluster_counts,
    x="Cluster",
    y="JumlahPelanggan",
    text="JumlahPelanggan",
    title="Jumlah Pelanggan pada Setiap Cluster"
)
st.plotly_chart(fig1, use_container_width=True)

st.subheader("Visualisasi Cluster")
x_axis = st.selectbox("Sumbu X", feature_cols, index=0)
y_axis = st.selectbox("Sumbu Y", feature_cols, index=2)

fig2 = px.scatter(
    filtered,
    x=x_axis,
    y=y_axis,
    color="Cluster",
    hover_data=["CustomerID"],
    title=f"{x_axis} vs {y_axis}"
)
st.plotly_chart(fig2, use_container_width=True)

st.subheader("Profil Cluster")
profile = (
    filtered.groupby("Cluster")[feature_cols]
    .mean()
    .round(2)
)
st.dataframe(profile, use_container_width=True)

st.subheader("Data Pelanggan")
st.dataframe(
    filtered.sort_values("Monetary", ascending=False),
    use_container_width=True
)

st.download_button(
    "⬇️ Download Hasil Segmentasi",
    data=filtered.to_csv(index=False).encode("utf-8"),
    file_name="customer_segments_filtered.csv",
    mime="text/csv"
)

st.sidebar.markdown("---")
st.sidebar.caption("Dataset: UCI Online Retail")
st.sidebar.caption("Metode: K-Means Clustering")
