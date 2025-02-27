# EEG Prediction API

## 🚀 Overview

This FastAPI-based API allows users to upload EEG data in `.parquet` format. The API processes the data, applies filtering, denoising, normalization, and other preprocessing steps, and then feeds it into a trained machine learning model to generate predictions.

## 📂 Features

* Accepts EEG data in `.parquet` format
* Preprocesses EEG data (bandpass filtering, wavelet denoising, normalization, variance & SNR checks)
* Converts processed EEG data into `.npy` format
* Runs inference using a trained model
* Returns prediction results via API

## 🛠️ Installation & Setup

### Prerequisites

Ensure you have the following installed:

* Python 3.8+
* `pip` (Python package manager)

### Clone the Repository

```sh
git clone https://github.com/yourusername/eeg-api.git
cd eeg-api
```

### Install Dependencies

```sh
pip install -r requirements.txt
```

### Run the API Locally

```sh
uvicorn main:app --reload
```

This starts the API on `http://127.0.0.1:8000`.

## 📡 API Endpoints

### 1️⃣ Upload EEG File

**Endpoint:** `POST /upload`

**Description:** Accepts a `.parquet` EEG file, processes it, and returns the shape of the processed data.

#### Request Example

```sh
curl -X 'POST' \
  'http://127.0.0.1:8000/upload' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@sample_eeg.parquet'
```

#### Response Example

```json
{
  "shape": [6000, 10]
}
```

## 🚀 Deployment Options

### Deploy on Render (Free & Easy)

1. Push your code to GitHub.
2. Go to [Render](https://render.com/).
3. Create a  **new Web Service** .
4. Connect your GitHub repo.
5. Set the start command:
   ```sh
   uvicorn main:app --host 0.0.0.0 --port $PORT
   ```
6. Deploy and get your public API URL!

### Alternative: Deploy on Railway

1. Sign up at [Railway](https://railway.app/).
2. Create a new project.
3. Deploy directly from GitHub.
4. Set the start command:
   ```sh
   uvicorn main:app --host 0.0.0.0 --port $PORT
   ```
5. Get your API running!

## 🛠️ Environment Variables

Create a `.env` file to store the required environment variables:

```env
MODEL_PATH=./models/trained_model.pth
BASE_PATH=./data
```

## 🧪 Running Tests

Run the test suite to ensure everything is working correctly:

```sh
pytest tests/
```

## 📜 License

This project is licensed under the MIT License.
