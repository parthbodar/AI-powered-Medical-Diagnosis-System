
# AI-Powered Medical Diagnosis System

## Overview
This project demonstrates an AI-powered medical diagnosis system that uses a deep learning model to predict diseases from medical images such as **X-rays** and **MRIs**. The model helps in detecting early signs of cancer or other critical conditions and assists radiologists in speeding up diagnosis with improved accuracy.

---

## Technologies Used
- **Python**
- **TensorFlow / Keras**: For building and loading the trained deep learning model.
- **OpenCV**: For image preprocessing.
- **Flask**: For creating the web API.
- **Postman**: For testing the API requests.
- **CNN (Convolutional Neural Network)**: To analyze the medical images.
- **Transfer Learning**: Used to improve the model performance with pre-trained networks.

---

## Features
1. **Upload medical images (X-rays or MRI scans)** via API for diagnosis.
2. **Predicts binary outcomes** (e.g., positive or negative) using deep learning.
3. **REST API** with Flask for easy integration into other systems.
4. **Error handling** for missing files and incorrect input.

---

## Project Structure

```
.
├── medical_diagnosis_model.h5   # Pre-trained model file
├── app.py                       # Flask app with API routes
├── uploaded_image.jpeg          # Temporary location to store uploaded images
├── requirements.txt             # List of dependencies
└── README.md                    # Documentation (this file)
```

---

## Setup and Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/medical-diagnosis-system.git
cd medical-diagnosis-system
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate    # On Linux/Mac
venv\Scripts\activate       # On Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Add the Model File
- Place your **`medical_diagnosis_model.h5`** file in the project directory.

---

## Running the Application

### 1. Start the Flask Server
```bash
python app.py
```
You should see the following output:
```
 * Running on http://127.0.0.1:5000
```

### 2. Test the API with Postman

#### POST /predict
- URL: `http://127.0.0.1:5000/predict`
- Method: **POST**
- Body: Use **form-data** and upload an image with the key name `image`.

**Expected Response:**
```json
{
    "prediction": "Positive"
}
```

#### GET /
- URL: `http://127.0.0.1:5000/`
- Method: **GET**
- Response:
```json
{
    "message": "API is running"
}
```

---

## Troubleshooting

1. **404 Error on /predict**
   - Ensure the route URL is **`http://127.0.0.1:5000/predict`** (no extra spaces or newlines).
   - Verify that the **POST method** is being used.

2. **500 Internal Server Error**
   - Ensure that the uploaded image is valid and readable.
   - Check if the **`medical_diagnosis_model.h5`** file is in the correct directory.
   - Verify that all dependencies are installed by running:
     ```bash
     pip install -r requirements.txt
     ```

3. **TensorFlow Warnings**
   - You can suppress the oneDNN optimization warning by setting:
     ```bash
     export TF_ENABLE_ONEDNN_OPTS=0  # On Linux/Mac
     set TF_ENABLE_ONEDNN_OPTS=0  # On Windows
     ```

---

## Requirements

Create a **`requirements.txt`** file with the following dependencies:

```
Flask==2.2.2
tensorflow==2.13.0
numpy==1.24.0
opencv-python==4.10.0
```

Install these dependencies using:

```bash
pip install -r requirements.txt
```

---

## Future Enhancements
1. **Add Support for Multiple Diagnoses** (multi-class classification).
2. **Improve the Model** using additional medical image datasets.
3. **Add Authentication** for secure access to the API.
4. **Deploy the Application** using a WSGI server (e.g., Gunicorn) or on platforms like **AWS**, **Heroku**, or **Azure**.

---

## License
This project is licensed under the **MIT License**. See the LICENSE file for more details.

---

## Contributing
Feel free to fork the repository and submit pull requests for new features or bug fixes.

---

## Contact
For any queries or feedback, please contact:
- **Your Name**: [your-email@example.com](mailto:your-email@example.com)
- GitHub: [https://github.com/your-username](https://github.com/your-username)

---

