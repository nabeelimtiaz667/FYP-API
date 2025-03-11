import os
from fastapi import FastAPI, File, UploadFile
import numpy as np
import pandas as pd
import pyarrow.parquet
import pywt
from scipy.signal import butter, filtfilt
import tensorflow as tf
import keras
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (change this in production)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Load trained model
MODEL_PATH = "cnn_lstm.keras"
model = keras.models.load_model(MODEL_PATH)

# Define constants
SELECTED_CHANNELS = ["Fp1", "O2", "T6", "Fz", "F4", "T3", "Cz", "T5", "C4", "P3"]
LOW_VARIANCE_THRESHOLD = 0.001
HIGH_VARIANCE_THRESHOLD = 10
EEG_SHAPE = (6000, 10, 1)

TRAN_CSV_PATH = "train.csv"  # Path to your expert consensus CSV file

# 🧠 Bandpass Filter (0.5–50 Hz)
def bandpass_filter(data, lowcut=0.5, highcut=50, fs=200, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, data, axis=0)


# 📉 Wavelet Denoising
def wavelet_denoise(data, wavelet="db4", level=2):
    coeffs = pywt.wavedec(data, wavelet, axis=0)
    threshold = np.median(np.abs(coeffs[-1])) / 0.6745
    coeffs = [pywt.threshold(c, threshold, mode="soft") for c in coeffs]
    return pywt.waverec(coeffs, wavelet, axis=0)


# 📊 Signal-to-Noise Ratio (SNR)
def calculate_snr(signal):
    mean_signal = np.mean(signal)
    noise = signal - mean_signal
    snr = 10 * np.log10(np.var(signal) / np.var(noise))
    return snr


# 🔄 Preprocess EEG Data
def preprocess_parquet(eeg_df):
    try:
        # Select Relevant Channels
        eeg_data = eeg_df[SELECTED_CHANNELS]

        # Replace NaN & Inf Values
        eeg_data = eeg_data.replace([np.inf, -np.inf], 0).fillna(0)

        # Apply Bandpass Filtering
        eeg_filtered = bandpass_filter(eeg_data)

        # Apply Wavelet Denoising
        eeg_denoised = wavelet_denoise(eeg_filtered)

        # Normalize Data
        stds = eeg_denoised.std(axis=0)
        stds[stds == 0] = 1e-8
        eeg_normalized = (eeg_denoised - eeg_denoised.mean(axis=0)) / stds
        eeg_normalized = np.nan_to_num(eeg_normalized)

        # Variance & SNR Checks
        variances = eeg_normalized.var(axis=0)
        if (variances < LOW_VARIANCE_THRESHOLD).any() or (
            variances > HIGH_VARIANCE_THRESHOLD
        ).any():
            return None
        snr_values = np.apply_along_axis(calculate_snr, axis=0, arr=eeg_normalized)
        if (snr_values < 0).any():
            return None

        # Extract First 6000 Samples (30 Sec Window)
        eeg_segment = eeg_normalized[:6000, :]

        return eeg_segment

    except Exception as e:
        print(f"Preprocessing Error: {e}")
        return None


# Function to get expert consensus from tran.csv
def get_expert_consensus(file_name: str):
    try:
        # Remove the file extension to match the eeg_id in tran.csv
        file_name_without_extension = os.path.splitext(file_name)[0]
        
        # Convert the file name (which is a string) to an integer to match the `eeg_id` type
        file_name_without_extension = int(file_name_without_extension)
        
        # Read the tran.csv file
        df = pd.read_csv(TRAN_CSV_PATH)

        # Ensure the `eeg_id` is also treated as an integer for comparison
        df['eeg_id'] = df['eeg_id'].astype(int)
        
        # Find the matching eeg_id for the file name (after stripping extension)
        consensus_row = df[df['eeg_id'] == file_name_without_extension]
        
        if not consensus_row.empty:
            return consensus_row['expert_consensus'].values[0]
        else:
            return f"No match is found. EEG ID: {file_name_without_extension}"  # If no match is found
    except Exception as e:
        print(f"Error while reading expert consensus: {e}")
        return f"Error while retrieving: {e}"



@app.post("/file-info/")
async def get_file_shape(file: UploadFile = File(...)):
    """
    Upload a file (.parquet or .npy), read it, and return its shape.
    """
    try:
        # Save file temporarily
        temp_filename = f"temp_{file.filename}"
        with open(temp_filename, "wb") as buffer:
            buffer.write(await file.read())

        # Determine file type and get shape
        if file.filename.endswith(".parquet"):
            df = pd.read_parquet(temp_filename)
            shape = df.shape  # (rows, columns)
        else:
            return {"error": "Unsupported file format. Use .npy or .parquet"}

        os.remove(temp_filename)

        # Preprocess EEG Data
        eeg_segment = preprocess_parquet(df)

        if eeg_segment is None:
            return {
                "error": "EEG file failed preprocessing checks (variance/SNR thresholds)."
            }

        # Reshape for Model Input (Add Channel & Batch Dimension)
        eeg_processed = np.expand_dims(
            eeg_segment, axis=-1
        )  # (6000, 10) → (6000, 10, 1)
        eeg_processed = np.expand_dims(eeg_processed, axis=0)  # Add batch dimension

        # Get Prediction
        prediction = model.predict(eeg_processed)
        predicted_class = np.argmax(prediction)  # Get class index
        confidence = float(np.max(prediction))  # Confidence Score

        # Convert Result
        result = "Seizure" if predicted_class == 1 else "Non-Seizure"

        # Get expert consensus from tran.csv
        real_prediction = get_expert_consensus(file.filename)

        return {
            "shape": shape,
            "prediction": result,
            "confidence": confidence,
            "real_prediction": real_prediction,
        }

    except Exception as e:
        return {"error": str(e)}
