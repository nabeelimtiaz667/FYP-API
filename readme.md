
# EEG Seizure Detection API & Streamlit App

## 🚀 Overview

This project integrates **FastAPI** and **Streamlit** to provide an end-to-end solution for **seizure detection** using EEG data. The backend API is built with FastAPI, which processes EEG data, applies preprocessing techniques, and runs inference using a pre-trained AI model. The frontend, developed with Streamlit, allows users to upload EEG data and visualize predictions interactively.

## 📂 Features

### FastAPI Backend:

* Accepts EEG data in `.parquet` format
* Preprocesses EEG data (bandpass filtering, wavelet denoising, normalization, variance & SNR checks)
* Runs inference using a **CNN-BiLSTM** model
* Provides seizure vs. non-seizure predictions via API
* Fetches expert consensus from a pre-labeled dataset

### Streamlit Frontend:

* User-friendly interface for uploading EEG files
* Displays **model predictions** and **confidence scores**
* Visualizes EEG spectrograms & dataset distributions
* Provides insights into **model training performance** across different folds

## 🔗 Project Links

* **Kaggle Notebook** : [Seizure Detection Model](https://www.kaggle.com/code/nabeel667/fyp-project-seizure-detection)
* **GitHub Repository** : [Seizure Detection API](https://github.com/nabeelimtiaz667/FYP-API)

## 🛠️ Installation & Setup

### Prerequisites

Ensure you have the following installed:

* Python 3.8+
* `pip` (Python package manager)

### Clone the Repository

```sh
git clone https://github.com/nabeelimtiaz667/FYP-API.git
cd FYP-API
```

### Install Dependencies

```sh
pip install -r requirements.txt
```

### Run the Application

You can launch the application using the provided executable:

```sh
launcher.exe
```

This will automatically start both the FastAPI backend and the Streamlit frontend, and open the application in your default web browser.

Alternatively, you can manually start the services:

#### Run the FastAPI Backend

```sh
uvicorn api:app --reload
```

This starts the API on `http://127.0.0.1:8000`.

#### Run the Streamlit App

```sh
streamlit run seizure_prediction_app.py
```

This starts the frontend where users can interact with the model.

## 📡 API Endpoints

### 1️⃣ Upload EEG File

**Endpoint:** `POST /file-info/`

**Description:** Accepts an EEG `.parquet` file, processes it, and returns predictions.

#### Request Example

```sh
curl -X 'POST' \
  'http://127.0.0.1:8000/file-info/' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@sample_eeg.parquet'
```

#### Response Example

```json
{
  "shape": [6000, 10],
  "prediction": "Seizure",
  "confidence": 0.95,
  "real_prediction": "Seizure"
}
```

## 🚀 Deployment Options

### Deploy on Render

1. Push your code to GitHub.
2. Go to [Render](https://render.com/).
3. Create a  **new Web Service** .
4. Connect your GitHub repo.
5. Set the start command:
   ```sh
   uvicorn api:app --host 0.0.0.0 --port $PORT
   ```
6. Deploy and get your public API URL!

### Alternative: Deploy on Railway

1. Sign up at [Railway](https://railway.app/).
2. Create a new project.
3. Deploy directly from GitHub.
4. Set the start command:
   ```sh
   uvicorn api:app --host 0.0.0.0 --port $PORT
   ```
5. Get your API running!

## 🛠️ Environment Variables

Create a `.env` file to store the required environment variables:

```env
MODEL_PATH=./cnn_lstm.keras
```

## 🧪 Running Tests

Run the test suite to ensure everything is working correctly:

```sh
pytest tests/
```

## 📜 License

This project is licensed under the MIT License.
