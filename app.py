import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
IMG_SIZE = 224
MODEL_PATH = r"busi_cnn_model.h5"
CLASS_NAMES = ["Normal", "Cancer"]
@st.cache_resource
def load_cnn_model():
    return load_model(MODEL_PATH)

model = load_cnn_model()
st.title("Breast Cancer Prediction")
st.write("Upload an ultrasound image")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
    img_reshaped = img_resized.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    prediction = model.predict(img_reshaped)
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = float(np.max(prediction)) * 100
    st.image(img, caption=f"Uploaded Image ({predicted_class}, {confidence:.2f}% confidence)", use_container_width=True)
    st.success(f"Prediction: **{predicted_class}** ({confidence:.2f}%)")
