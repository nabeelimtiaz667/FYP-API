EEG Seizure Detection API & Streamlit App

=======================================

OVERVIEW
---------
This project integrates FastAPI and Streamlit to provide an end-to-end solution for seizure detection using EEG data. The backend API is built with FastAPI, which processes EEG data, applies preprocessing techniques, and runs inference using a pre-trained AI model. The frontend, developed with Streamlit, allows users to upload EEG data and visualize predictions interactively.

FEATURES
--------
FastAPI Backend:
- Accepts EEG data in `.parquet` format
- Preprocesses EEG data (bandpass filtering, wavelet denoising, normalization, variance & SNR checks)
- Runs inference using a CNN-BiLSTM model
- Provides seizure vs. non-seizure predictions via API
- Fetches expert consensus from a pre-labeled dataset

Streamlit Frontend:
- User-friendly interface for uploading EEG files
- Displays model predictions and confidence scores
- Visualizes EEG spectrograms & dataset distributions
- Provides insights into model training performance across different folds

PROJECT LINKS
-------------
- Kaggle Notebook: https://www.kaggle.com/code/nabeel667/fyp-project-seizure-detection
- GitHub Repository: https://github.com/nabeelimtiaz667/FYP-API

INSTALLATION & SETUP
----------------------
Prerequisites:
- Python 3.8+
- pip (Python package manager)

Clone the Repository:
1. git clone https://github.com/nabeelimtiaz667/FYP-API.git
2. cd FYP-API

Install Dependencies:
- pip install -r requirements.txt

Run the Application:
You can launch the application using the provided executable:
- launcher.exe
This will automatically start both the FastAPI backend and the Streamlit frontend, and open the application in your default web browser.

Alternatively, manually start the services:
Run the FastAPI Backend:
- uvicorn api:app --reload
This starts the API on http://127.0.0.1:8000.

Run the Streamlit App:
- streamlit run seizure_prediction_app.py

API ENDPOINTS
-------------
Upload EEG File:
- Endpoint: POST /file-info/
- Description: Accepts an EEG `.parquet` file, processes it, and returns predictions.

Example Request:
```
curl -X 'POST' \
  'http://127.0.0.1:8000/file-info/' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@sample_eeg.parquet'
```

Example Response:
```
{
  "shape": [6000, 10],
  "prediction": "Seizure",
  "confidence": 0.95,
  "real_prediction": "Seizure"
}
```

DEPLOYMENT OPTIONS
-------------------
Deploy on Render:
1. Push your code to GitHub.
2. Go to https://render.com/.
3. Create a new Web Service.
4. Connect your GitHub repo.
5. Set the start command:
   - uvicorn api:app --host 0.0.0.0 --port $PORT
6. Deploy and get your public API URL!

Deploy on Railway:
1. Sign up at https://railway.app/.
2. Create a new project.
3. Deploy directly from GitHub.
4. Set the start command:
   - uvicorn api:app --host 0.0.0.0 --port $PORT
5. Get your API running!

ENVIRONMENT VARIABLES
----------------------
Create a `.env` file to store the required environment variables:
```
MODEL_PATH=./cnn_lstm.keras
BASE_PATH=./data
```

RUNNING TESTS
--------------
Run the test suite to ensure everything is working correctly:
- pytest tests/

LICENSE
--------
This project is licensed under the MIT License.

