import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

MODEL_PATH = "model/model.tflite"
CLASS_PATH = "model/class_names.json"
SAMPLE_DIR = "Skin_images"
IMG_SIZE = (224, 224)

st.set_page_config(
    page_title="Skin Cancer Detection",
    page_icon="🧬",
    layout="centered"
)

@st.cache_resource
def load_interpreter():
    try:
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"Failed to load TFLite model: {e}")
        st.stop()

@st.cache_data
def load_class_names():
    try:
        with open(CLASS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Failed to load class names: {e}")
        st.stop()

def get_label(class_names, index):
    if isinstance(class_names, list):
        return class_names[index]
    if isinstance(class_names, dict):
        return class_names.get(str(index), f"Class {index}")
    return f"Class {index}"

def preprocess_image(image: Image.Image, interpreter):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)

    input_details = interpreter.get_input_details()
    input_dtype = input_details[0]["dtype"]

    arr = np.array(image)

    if input_dtype == np.float32:
        arr = arr.astype(np.float32) / 255.0
    else:
        arr = arr.astype(input_dtype)

    arr = np.expand_dims(arr, axis=0)
    return arr

def predict_tflite(interpreter, input_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]["index"])
    return output_data

def postprocess_predictions(raw_output):
    raw_output = np.array(raw_output, dtype=np.float32)

    if (
        np.any(raw_output < 0)
        or np.any(raw_output > 1)
        or not np.isclose(np.sum(raw_output), 1.0, atol=1e-2)
    ):
        exp_vals = np.exp(raw_output - np.max(raw_output))
        raw_output = exp_vals / np.sum(exp_vals)

    return raw_output

def run_prediction(image: Image.Image, interpreter):
    input_image = preprocess_image(image, interpreter)
    raw_predictions = predict_tflite(interpreter, input_image)[0]
    predictions = postprocess_predictions(raw_predictions)

    predicted_index = int(np.argmax(predictions))
    confidence = float(predictions[predicted_index]) * 100

    return predictions, predicted_index, confidence

def show_results(image: Image.Image, predictions, predicted_index, confidence, class_names):
    st.image(image, caption="Selected Image")

    label = get_label(class_names, predicted_index)

    st.subheader(f"Prediction: {label}")
    st.info(f"Confidence: {confidence:.2f}%")

    st.subheader("Top 3 Predictions")
    top3_indices = np.argsort(predictions)[-3:][::-1]

    for i in top3_indices:
        st.write(f"{get_label(class_names, i)}: {predictions[i] * 100:.2f}%")

    st.subheader("Prediction Probabilities")
    chart_df = pd.DataFrame(
        {"Probability (%)": predictions * 100},
        index=[get_label(class_names, i) for i in range(len(predictions))]
    )
    chart_df = chart_df.sort_values("Probability (%)", ascending=False)
    st.bar_chart(chart_df)

    st.warning(
        "This AI prediction is for educational purposes only and does not replace medical diagnosis."
    )

st.markdown(
    """
    <h1 style='text-align: center; color: #4CAF50;'>
    🧬 Skin Cancer Detection using CNN
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    This application uses a **Convolutional Neural Network (CNN)** to classify skin lesion images.

    You can either:
    - upload your own skin image
    - or test the model using sample images
    """
)

interpreter = load_interpreter()
class_names = load_class_names()

tab1, tab2 = st.tabs(["Upload Image", "Sample Images"])

with tab1:
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        with st.spinner("Running prediction..."):
            predictions, predicted_index, confidence = run_prediction(image, interpreter)

        show_results(image, predictions, predicted_index, confidence, class_names)

with tab2:
    if os.path.exists(SAMPLE_DIR):
        sample_images = sorted(
            [
                f for f in os.listdir(SAMPLE_DIR)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
        )

        if sample_images:
            selected_image = st.selectbox("Choose a sample image", sample_images)

            if st.button("Run prediction on sample image"):
                image_path = os.path.join(SAMPLE_DIR, selected_image)
                image = Image.open(image_path).convert("RGB")

                with st.spinner("Running prediction..."):
                    predictions, predicted_index, confidence = run_prediction(image, interpreter)

                show_results(image, predictions, predicted_index, confidence, class_names)
        else:
            st.warning("No sample images found inside the Skin_images folder.")
    else:
        st.warning("Skin_images folder was not found.")

st.markdown("---")
st.markdown(
    """
    ### About the Model
    - Model type: CNN
    - Deployment format: TensorFlow Lite
    - Task: Multi-class skin lesion classification

    **Note:** This project is intended for educational and research purposes only.
    """
)