import tensorflow as tf
import numpy as np
import cv2
import os

# Load the Trained Model
model = tf.keras.models.load_model('medical_diagnosis_model.h5')

def predict_image(image_path):
    # Check if the file exists
    if not os.path.exists(image_path):
        print(f"Error: File not found at {image_path}")
        return

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from path {image_path}")
        return

    # Preprocess the image
    image = cv2.resize(image, (224, 224)) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Make Prediction
    prediction = model.predict(image)[0][0]
    result = "Positive" if prediction >= 0.5 else "Negative"
    return result

# Test the Prediction Function
image_path = 'IM-0036-0001.jpeg'
print(f"Prediction: {predict_image(image_path)}")
