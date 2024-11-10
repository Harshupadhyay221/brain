import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('brain_tumor_classifier.h5')

# Class names for the 4 categories
CLASS_NAMES = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']  # Replace with your actual class names

# Function to preprocess uploaded images
def preprocess_image(image):
    # Use Image.Resampling.LANCZOS instead of Image.ANTIALIAS
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)  # Resize image to 224x224
    img_array = np.asarray(image)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize
    return img_array

# Prediction function
def predict(image):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]  # Get the predicted class
    confidence = np.max(predictions)  # Get the confidence score
    return predicted_class, confidence

# Streamlit interface
st.title("Brain Tumor Classification")

st.write("""
Upload an MRI image, and the model will classify it into one of four tumor types.
""")

# File uploader for MRI image
uploaded_file = st.file_uploader("Choose an MRI image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded MRI Image', use_column_width=True)

    if st.button("Classify"):
        with st.spinner('Classifying...'):
            predicted_class, confidence = predict(image)
            st.write(f"**Predicted Class:** {predicted_class}")
            st.write(f"**Confidence:** {confidence * 100:.2f}%")
