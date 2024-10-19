from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2

# 1. Load the Trained Model
model = tf.keras.models.load_model('medical_diagnosis_model.h5')

app = Flask(__name__)

# Define a basic route for '/'
@app.route('/')
def home():
    return jsonify({"message": "API is running"}), 200

# Handle favicon.ico requests to avoid unnecessary 404 errors
@app.route('/favicon.ico')
def favicon():
    return '', 204

# Prediction Route
@app.route('/predict', methods=['POST'])
def predict():
    # Log request details
    print("Received request to /predict")

    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    image_path = 'uploaded_image.jpeg'
    file.save(image_path)

    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Failed to load the image.")
        image = cv2.resize(image, (224, 224)) / 255.0
        image = np.expand_dims(image, axis=0)

        prediction = model.predict(image)[0][0]
        result = "Positive" if prediction >= 0.5 else "Negative"

        return jsonify({'prediction': result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask App
if __name__ == '__main__':
    app.run(debug=True)
