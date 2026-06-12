# app.py

import streamlit as st
from sklearn import datasets
from sklearn.model_selection import train_test_split
from scrappy_knn import ScrappyKNN

# Load dataset
iris = datasets.load_iris()
features = iris.data
labels = iris.target

# Split dataset
features_train, features_test, labels_train, labels_test = train_test_split(
    features, labels, test_size=0.5, random_state=42
)

# Train model
model = ScrappyKNN()
model.fit(features_train, labels_train)

# Page Config
st.set_page_config(
    page_title="ScrappyKNN Iris Predictor",
    page_icon="🌸",
    layout="centered"
)

# Title
st.title("🌸 ScrappyKNN Iris Predictor")
st.write("Predict Iris flower species using your custom ScrappyKNN algorithm.")

# Input Fields
sepal_length = st.number_input(
    "Sepal Length (cm)",
    min_value=0.0,
    value=5.1
)

sepal_width = st.number_input(
    "Sepal Width (cm)",
    min_value=0.0,
    value=3.5
)

petal_length = st.number_input(
    "Petal Length (cm)",
    min_value=0.0,
    value=1.4
)

petal_width = st.number_input(
    "Petal Width (cm)",
    min_value=0.0,
    value=0.2
)

# Prediction Button
if st.button("Predict Species"):
    sample = [[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]]

    prediction = model.predict(sample)

    species_map = {
        0: "Setosa",
        1: "Versicolor",
        2: "Virginica"
    }

    species = species_map[prediction[0]]

    st.success(f"Predicted Species: {species}")
    st.write(f"Class ID: {prediction[0]}")