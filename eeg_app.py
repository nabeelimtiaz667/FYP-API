import streamlit as st
import requests
from io import BytesIO

# Define FastAPI URL (local or deployed)
API_URL = "http://127.0.0.1:8000/file-info/"

# Title of the Streamlit app
st.title("EEG Seizure Prediction")

# File Upload Section
uploaded_file = st.file_uploader("Upload a .parquet or .npy file", type=["parquet", "npy"])

if uploaded_file is not None:
    # Display the file name and type
    st.write(f"File name: {uploaded_file.name}")
    
    # Prepare the file for API call
    files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
    
    # Call FastAPI endpoint
    try:
        # Make a POST request to the FastAPI server with the file
        response = requests.post(API_URL, files=files)

        # Parse the response
        if response.status_code == 200:
            data = response.json()
            
            if "error" in data:
                st.error(data["error"])
            else:
                # Show the file shape
                st.write(f"File Shape: {data['shape']}")
                
                # Show the model prediction result and confidence
                prediction = data['prediction']
                confidence = data['confidence']
                st.write(f"Prediction: {prediction}")
                st.write(f"Confidence: {confidence:.2f}")
                
                # Show the real prediction (expert consensus)
                real_prediction = data['real_prediction']
                st.write(f"Real Prediction (Expert Consensus): {real_prediction}")
                
        else:
            st.error("Error: Unable to get the result from the FastAPI server.")
    
    except Exception as e:
        st.error(f"An error occurred: {e}")
