import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import datetime

# Load the trained model
model = tf.keras.models.load_model("best_model.h5")

# Apply custom CSS styling
st.markdown(
    """
    <style>
    /* Background and text color */
    .main {
        background-color: black;
        color: white;
    }
    
    /* Title styling */
    h1 {
        color: #FFD700;  /* Gold */
        text-align: center;
        font-size: 32px;
        font-weight: bold;
    }

    /* File uploader styling */
    div.stFileUploader {
        border-radius: 10px;
        border: 2px solid #FFD700;
        padding: 10px;
        background-color: #222222;
        color: white;
        font-size: 16px;
    }

    /* Button styling */
    div.stButton>button {
        background-color: #FFD700;
        color: black;
        font-size: 18px;
        border-radius: 10px;
        padding: 10px;
        border: none;
        cursor: pointer;
    }

    /* Prediction result styling */
    .prediction {
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        color: #FFD700;
    }

    /* Confidence level styling */
    .confidence {
        text-align: center;
        font-size: 20px;
        color: white;
    }

    /* Report styling */
    .report {
        background-color: #222222;
        color: white;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #FFD700;
        font-size: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Function to preprocess image
def preprocess_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  # Read as RGB
    
    # Resize and normalize
    img = cv2.resize(img, (150, 150)) / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    
    return img

# Streamlit UI
st.markdown('<h1>🌟 Pneumonia Detection 🌟</h1>', unsafe_allow_html=True)
st.write("_Upload a chest X-ray to predict whether it is **Normal** or **Pneumonia**, along with confidence level._")

uploaded_file = st.file_uploader("📂 Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = preprocess_image(uploaded_file)  # Preprocess image
    prediction_prob = model.predict(img)[0][0]  # Get probability
    confidence = round(prediction_prob * 100, 2)  # Convert to percentage

    result = "🩺 Pneumonia" if prediction_prob > 0.5 else "✅ Normal"

    # Display uploaded image
    st.image(uploaded_file, caption="🖼 Uploaded Image", use_column_width=True)
    
    # Show prediction result with confidence score
    st.markdown(f'<div class="prediction"> Prediction: {result} </div>', unsafe_allow_html=True)
    st.markdown(f'<div class="confidence"> Confidence Level: {confidence}% </div>', unsafe_allow_html=True)

    # Generate report
    st.subheader("📜 Prediction Report")
    report = f"""
    <div class="report">
    **Pneumonia Detection Report**
    ------------------------------
    **Date:** {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    
    **Prediction:** {result}
    **Confidence Level:** {confidence}%
    
    **Uploaded Image:** Chest X-ray
    
    **Model:** CNN-based Pneumonia Detection
    </div>
    """
    st.markdown(report, unsafe_allow_html=True)
