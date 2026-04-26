import streamlit as st
import pickle
import numpy as np

# Load model & scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Flower images
flower_images = {
    'Iris-setosa': './iris-sentosa.jpg',
    'Iris-versicolor': './Iris-versicolor.jpg',
    'Iris-virginica': './Iris-virginica.jpg'
}

# Title
st.title("🌸 Iris Flower Prediction App")
st.write("Enter flower measurements:")

# Inputs
sepal_length = st.number_input("Sepal Length", min_value=0.0, value=5.0)
sepal_width = st.number_input("Sepal Width", min_value=0.0, value=3.0)
petal_length = st.number_input("Petal Length", min_value=0.0, value=4.0)
petal_width = st.number_input("Petal Width", min_value=0.0, value=1.0)

# Predict
if st.button("Predict"):
    
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    features = scaler.transform(features)
    
    prediction = model.predict(features)[0]
    flower = str(prediction)
    
    st.success(f"🌼 Predicted Flower: {flower}")
    st.image(flower_images[flower], caption=flower, width=300)