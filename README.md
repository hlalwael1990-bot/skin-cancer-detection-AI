# Skin Cancer Detection using CNN

This project is a deep learning-based skin lesion image classifier built with TensorFlow and Streamlit.

## Features
- Upload a skin lesion image
- Predict the lesion class
- Show confidence score
- Display top 3 predictions
- Test with sample images
- Visualize prediction probabilities

## Model
The model was trained on a skin lesion dataset and exported as a TensorFlow Lite model for deployment.

## Project Structure
```text
skin-cancer-classifier/
│
├── model/
│   ├── model.tflite
│   └── class_names.json
│
├── Skin_images/
│   ├── sample1.jpg
│   ├── sample2.jpg
│   └── ...
│
├── app.py
├── requirements.txt
├── runtime.txt
└── README.md