import subprocess
import webbrowser
import time


# Function to start FastAPI
def start_fastapi():
    # Run the FastAPI server using subprocess
    subprocess.Popen(["uvicorn", "api:app", "--host", "127.0.0.1", "--port", "8000"])


# Function to start Streamlit
def start_streamlit():
    subprocess.run(["streamlit", "run", "seizure_prediction_app.py"])


# Start FastAPI backend in a separate subprocess
start_fastapi()

# Give FastAPI some time to start before launching Streamlit
time.sleep(10)

# Start Streamlit frontend
start_streamlit()

# Optionally, open the Streamlit app in the default browser
webbrowser.open("http://localhost:8501")
