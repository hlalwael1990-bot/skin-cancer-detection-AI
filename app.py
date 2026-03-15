import os
import json
import numpy as np
import streamlit as st
from PIL import Image
from tflite_runtime.interpreter import Interpreter

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
    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter


@st.cache_data
def load_class_names():
    with open(CLASS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    arr = np.array(image, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


def predict_tflite(interpreter, input_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]["index"])
    return output_data


def run_prediction(image: Image.Image, interpreter):
    input_image = preprocess_image(image)
    predictions = predict_tflite(interpreter, input_image)

    predicted_index = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][predicted_index]) * 100

    return predictions[0], predicted_index, confidence


def show_results(image: Image.Image, predictions, predicted_index, confidence, class_names):
    st.image(image, caption="Selected Image", use_container_width=True)

    label = class_names[predicted_index]

    if "melanoma" in label.lower():
        st.error(f"Prediction: {label}")
    else:
        st.success(f"Prediction: {label}")

    st.info(f"Confidence: {confidence:.2f}%")

    st.subheader("Top 3 Predictions")
    top3_indices = np.argsort(predictions)[-3:][::-1]

    for i in top3_indices:
        st.write(f"{class_names[i]}: {predictions[i] * 100:.2f}%")

    st.subheader("Prediction Probabilities")

    sorted_indices = np.argsort(predictions)[::-1]
    for i in sorted_indices:
        prob = float(predictions[i]) * 100
        st.write(f"{class_names[i]}: {prob:.2f}%")
        st.progress(min(int(prob), 100))

    st.warning(
        "This AI prediction is for educational purposes only and does not replace medical diagnosis."
    )

    if st.button("Clear Result"):
        st.rerun()


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
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        with st.spinner("Running prediction..."):
            predictions, predicted_index, confidence = run_prediction(
                image, interpreter
            )

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
                    predictions, predicted_index, confidence = run_prediction(
                        image, interpreter
                    )

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