import streamlit as st
import joblib
import numpy as np

# Page config
st.set_page_config(page_title="Customer Segmentation", layout="wide")

# Title
st.title("Customer Segmentation using K-Means")

st.write("Enter customer details to predict which segment they belong to.")

# Load model
kmeans = joblib.load("models/kmeans_model.joblib")
scaler = joblib.load("models/scaler.joblib")

# Sidebar input
st.sidebar.header("Input Features")

age = st.sidebar.slider("Age", 18, 70, 30)
income = st.sidebar.slider("Annual Income (k$)", 10, 150, 50)
spending = st.sidebar.slider("Spending Score (1-100)", 1, 100, 50)

# Predict
input_data = np.array([[age, income, spending]])
input_scaled = scaler.transform(input_data)

cluster = kmeans.predict(input_scaled)[0]

# Cluster meaning (🔥 adds value)
cluster_meaning = {
    0: "Low Income, Low Spending",
    1: "High Income, High Spending",
    2: "Medium Income, Medium Spending",
    3: "High Income, Low Spending",
    4: "Low Income, High Spending"
}

# Result
st.subheader("Prediction Result")
st.success(f"This customer belongs to Cluster {cluster}")

# Show interpretation
if cluster in cluster_meaning:
    st.info(f"Segment Description: {cluster_meaning[cluster]}")

# Visualizations
st.subheader("Visualizations")

col1, col2 = st.columns(2)

with col1:
    st.image("plots/clusters.png", caption="Customer Segments")

with col2:
    st.image("plots/elbow_method.png", caption="Elbow Method")

st.image("plots/pca_clusters.png", caption="PCA Cluster Visualization")

# Footer (🔥 PROFESSIONAL TOUCH)
st.markdown("---")

st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <h4>Customer Segmentation App</h4>
        <p>Built using Machine Learning (K-Means Clustering)</p>
        <p><b>Made by Lalitaditya Tickoo</b></p>
    </div>
    """,
    unsafe_allow_html=True
)