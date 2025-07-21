import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load the transfer learning model
model = load_model('models/transfer_learning_model.h5')
class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

# Streamlit app
st.title("Brain Tumor MRI Classification")
st.write("Upload an MRI image to classify it as glioma, meningioma, no tumor, or pituitary.")

# File uploader
uploaded_file = st.file_uploader("Choose an MRI image...", type=['jpg', 'png'])

if uploaded_file is not None:
    # Load and preprocess image
    image = Image.open(uploaded_file).convert('RGB')
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Predict
    prediction = model.predict(image_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # Display results
    st.image(image, caption='Uploaded MRI Image', use_column_width=True)
    st.write(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}%")