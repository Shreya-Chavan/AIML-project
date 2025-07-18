import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
from PIL import Image
import cv2

# Load the trained model
model = load_model('breast_cancer_cnn_model.h5')

# Title and UI
st.title("🧠 Breast Cancer Prediction System")
st.write("Upload a histopathology or mammogram image to check for cancer prediction.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img) 
    img_array = np.expand_dims(img_array, axis=0) 

    # Make prediction
    prediction = model.predict(img_array) [0][0]
    confidence = prediction if prediction > 0.5 else 1 - prediction
    label = "Malignant" if prediction > 0.5 else "Benign"
    
    # Show result
    st.write(f"**Prediction:** {label}")
    st.markdown(f"**Confidence:** {confidence*100:.2f}%")
