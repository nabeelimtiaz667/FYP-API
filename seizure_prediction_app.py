import streamlit as st
import requests
import matplotlib.pyplot as plt
import os
import subprocess
from PIL import Image

# Define FastAPI URLs
API_URL = "http://127.0.0.1:8000/file-info/"
VISUALS_URL = "http://127.0.0.1:8000/get-visuals/"

# Set App Title
st.set_page_config(page_title="EEG Seizure Prediction", layout="wide")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Our Project", "Prediction", "Model Visualization"])

if page == "Our Project":
    # Project Title
    st.markdown(
        "<h1 style='text-align: center; color: #1E88E5;'>EEG Seizure Detection & Prediction</h1>",
        unsafe_allow_html=True,
    )

    st.markdown("---")  # Separator Line

    # Introduction
    st.subheader("📌 Introduction")
    st.markdown(
        """
        - This project is focused on detecting **seizures** using EEG (Electroencephalography) signals.
        - We developed a **Deep Learning** pipeline to classify EEG signals and predict seizures.
        - The model is trained on EEG spectrograms using **Convolutional Neural Networks (CNNs)** and **Bidirectional LSTMs**.
        - Our solution aims to assist **neurologists** in **early seizure detection** and **automated analysis**.
        """
    )

    st.markdown("---")

    # Dataset Information
    st.subheader("📊 Dataset Information")
    st.markdown(
        """
        - The dataset contains EEG signals recorded under the **10-20 system** at **200 Hz**.
        - We have EEG data in `.parquet` format and spectrogram representations in `.npy` format.
        - The dataset consists of **multiple EEG channels** and **seizure annotations**.
        - The **main classes** include:
            - **Seizure**
            - **GRDA (Generalized Rhythmic Delta Activity)**
            - **GPD (Generalized Periodic Discharges)**
            - **LRDA (Lateralized Rhythmic Delta Activity)**
            - **LPD (Lateralized Periodic Discharges)**
            - **Other (Normal EEG signals)**
        """
    )

    st.markdown("---")

    # Model Architecture
    st.subheader("🧠 Model Architecture")
    st.markdown(
        """
        - We designed a **CNN-BiLSTM** model for feature extraction and sequence learning.
        - **Convolutional layers** capture spatial patterns in EEG spectrograms.
        - **Bidirectional LSTM layers** learn temporal dependencies in EEG signals.
        - The model was trained using the **Adam Optimizer** and **categorical cross-entropy loss**.
        - **Key Components:**
            - **Input:** EEG spectrogram (converted from raw EEG)
            - **Feature Extraction:** CNN layers with ReLU activation
            - **Temporal Analysis:** BiLSTM layers
            - **Classification:** Fully connected (Dense) layers with Softmax activation
        """
    )

    st.markdown("---")

    # Results
    st.subheader("📈 Results & Performance")
    st.markdown(
        """
        - Our model achieved **high accuracy** in seizure prediction with:
            - **Specificity:** 95%
            - **Precision:** 95%
            - **Recall:** 93%
            - **F1-Score:** 94%
        - **Key Findings:**
            - Seizures have distinct patterns in EEG spectrograms.
            - The CNN-BiLSTM model performed better than standard CNNs.
            - Expert-labeled data was essential for improving reliability.
        """
    )

    # Displaying Metrics
    col1, col2, col3 = st.columns(3)
    col3.metric("Precision", "95%")
    col1.metric("Sensitivity", "93%")
    col2.metric("Specificity", "95%")

    st.markdown("---")

    # Closing Statement
    st.markdown(
        """
        🚀 This project demonstrates **how AI can assist in medical diagnostics**. Future work involves:
        - **Expanding the dataset** with more EEG records.
        - **Improving the model** with attention mechanisms.
        - **Deploying a real-time seizure detection system** in hospitals.
        """
    )

    st.markdown("---")

    # Kaggle Notebook Link
    st.subheader("📖 Kaggle Notebook")
    st.markdown(
        "You can explore our Kaggle Notebook for more details and experiments: "
        "[EEG Seizure Detection & Prediction Notebook](https://www.kaggle.com/code/nabeel667/fyp-project-seizure-detection)",
        unsafe_allow_html=True,
    )

elif page == "Prediction":
    st.header("EEG Seizure Prediction")
    uploaded_file = st.file_uploader(
        "Upload a .parquet or .npy file", type=["parquet", "npy"]
    )

    if uploaded_file is not None:
        st.write(f"File name: {uploaded_file.name}")
        files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}

        try:
            response = requests.post(API_URL, files=files)
            if response.status_code == 200:
                data = response.json()

                if "error" in data:
                    st.error(data["error"])
                else:
                    st.write(f"File Shape: {data['shape']}")
                    st.write(f"Prediction: {data['prediction']}")
                    st.write(f"Confidence: {data['confidence']:.2f}")
                    st.write(
                        f"Real Prediction (Expert Consensus): {data['real_prediction']}"
                    )
            else:
                st.error("Error: Unable to get the result from the FastAPI server.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

elif page == "Model Visualization":
    st.header("Model Visualization")
    st.markdown(
        """
        This section provides insights into the dataset and model performance across different training folds.
        Below are visualizations and analyses that highlight key patterns in the EEG spectrogram dataset 
        and how well the deep learning model performed during training.
        """
    )
    # Dataset Visualizations
    st.subheader("📊 Dataset Visualizations")
    st.image(
        "visuals/dataset_visual_1.png",
        caption="Feature Distributions",
        use_container_width=True,
    )
    st.markdown(
        """
        - This visualization represents feature distributions within the EEG dataset.
        - Some features exhibit structured trends, while others show high noise levels.
        - Understanding these patterns is crucial for data preprocessing and feature selection.
        """
    )
    st.image(
        "visuals/dataset_visual_2.png",
        caption="Spectrogram Representation",
        use_container_width=True,
    )
    st.markdown(
        """
        - This figure likely showcases EEG spectrogram data or frequency-based patterns.
        - Fluctuations suggest time-dependent characteristics, emphasizing the need for temporal analysis.
        - Features extracted from these representations are used as inputs to the deep learning model.
        """
    )
    st.markdown("---")

    # Model Performance Across Folds
    st.subheader("📈 Model Performance Across Folds")
    fold_images = {
        "Fold 1": "visuals/fold_1.png",
        "Fold 2": "visuals/fold_2.png",
        "Fold 3": "visuals/fold_3.png",
        "Fold 4": "visuals/fold_4.png",
        "Fold 5": "visuals/fold_5.png",
    }
    fold_analysis = {
        "Fold 1": """
        ## **Fold 1:**
### **Training Loss & Validation Loss:**
- Training loss **steadily decreases** with each epoch.
- Validation loss fluctuates slightly but follows a downward trend.
- No drastic jumps, indicating stable training.

### **Training Accuracy & Validation Accuracy:**
- Accuracy **improves progressively**, peaking before early stopping.
- Validation accuracy has **minor fluctuations**, suggesting slight overfitting or variability in validation data.

### **Observations:**
- **Generalization:** Fairly good, though minor instability in validation performance.
- **Training Efficiency:** Completed full 10 epochs, meaning it did not converge too early.
- **Possible Concern:** Slight overfitting signs but nothing severe.
        """,
        "Fold 2": """
        ## **Fold 2:**
### **Training Loss & Validation Loss:**
- Both losses **decrease smoothly** over epochs.
- **Minimal fluctuation** in validation loss, suggesting better stability than Fold 1.

### **Training Accuracy & Validation Accuracy:**
- Accuracy **steadily increases** without abrupt changes.
- Validation accuracy aligns closely with training accuracy, a sign of strong generalization.

### **Observations:**
- **Generalization:** Excellent, with minimal divergence between training and validation performance.
- **Training Efficiency:** Reached 10 epochs, meaning no rapid early convergence.
- **Stability:** One of the most stable folds in training.
        """,
        "Fold 3": """
        ## **Fold 3:**
### **Training Loss & Validation Loss:**
- Loss **drops sharply** in the first few epochs.
- Early stopping activated around epoch **3**, meaning validation loss stopped improving beyond that point.

### **Training Accuracy & Validation Accuracy:**
- Accuracy **jumps quickly** to a high value (~91%) in just 3 epochs.
- Validation accuracy **matches closely** with training accuracy.

### **Observations:**
- **Early Stopping:** This fold converged too fast, meaning it likely had easier data.
- **Generalization:** Good, but rapid convergence might indicate the model didn't explore much beyond initial learning.
- **Potential Issue:** Might be overfitting slightly due to quick accuracy jumps.
        """,
        "Fold 4": """
        ## **Fold 4:**
### **Training Loss & Validation Loss:**
- Loss **fluctuates initially**, stabilizing only in later epochs.
- Validation loss shows a **spike**, indicating a noisy dataset or difficult samples.

### **Training Accuracy & Validation Accuracy:**
- Accuracy **drops initially**, then recovers, unlike the smooth trends in other folds.
- Validation accuracy **mirrors this instability**, meaning this fold likely had **challenging validation data**.

### **Observations:**
- **Generalization:** Weaker than previous folds due to instability.
- **Training Efficiency:** Required full 10 epochs to stabilize.
- **Potential Issue:** May indicate a harder-to-learn subset of data.
        """,
        "Fold 5": """
        ## **Fold 5:**
### **Training Loss & Validation Loss:**
- Loss **declines steadily** with very little fluctuation.
- Early stopping triggered at **epoch 3**, meaning fast convergence.

### **Training Accuracy & Validation Accuracy:**
- Accuracy **rises rapidly**, peaking at ~95% within 3 epochs.
- Validation accuracy follows training accuracy closely.

### **Observations:**
- **Early Stopping:** Suggests an **easier dataset** within this fold.
- **Generalization:** Strong, with no major overfitting signs.
- **Best Performing Fold:** This fold had the **fastest and most stable training**.
        """,
    }
    for fold, img_path in fold_images.items():
        st.image(
            img_path, caption=f"{fold} Training Performance", use_container_width=True
        )
        analysis = fold_analysis[fold]
        st.markdown(analysis)
        st.markdown("---")
    st.markdown(
        """
                ## **Final Summary:**
                | **Fold** | **Early Stopping?** | **Generalization** | **Training Stability** | **Potential Issue** |
                |---------|-----------------|----------------|------------------|-----------------|
                | **1** | No (10 epochs) | Good | Slight validation fluctuations | Minor overfitting |
                | **2** | No (10 epochs) | Excellent | Very stable | None |
                | **3** | Yes (3 epochs) | Good | Fast convergence | Possible overfitting |
                | **4** | No (10 epochs) | Moderate | Unstable at start | Harder dataset |
                | **5** | Yes (3 epochs) | Excellent | Very stable | None |

                ### **Key Takeaways:**
                1. **Folds 3 & 5 Converged Extremely Fast:** Indicating easy data samples.
                2. **Fold 4 Was the Most Challenging:** Suggesting harder-to-learn data.
                3. **Fold 2 Had the Best Balance:** Generalized well with stable validation.
                4. **Early Stopping Helped Prevent Overfitting in Folds 3 & 5.**
                ---
                """
    )

    # Summary Insights
    st.subheader("🔍 Summary & Insights")
    st.markdown(
        """
        - **EEG Dataset Observations:**
            - The dataset contains both structured and noisy signals, impacting learning efficiency.
            - EEG-based features effectively capture temporal brain activity but may require additional preprocessing to enhance stability.
        - **Model Performance Trends:**
            - The model generally learned well across folds, but certain folds (Fold 4) posed challenges, leading to fluctuations in validation loss.
            - Fast convergence in Folds 3 & 5 suggests some subsets may have been easier to classify, highlighting potential class imbalances.
        - **What was our Approach:**
            - Create **data preprocessing pipeline** by applying bandpass filters, wavelet denoising, and calculating signal-to-noise ratio.
            - Implemented **data augmentation** to introduce variations in EEG signals, reducing reliance on easy samples.
            - Applied **regularization methods** (e.g., dropout, weight decay) to prevent overfitting in fast-converging folds.
            - Fine-tuned **early stopping patience** to ensure each fold gets optimal training duration.
        """
    )
    st.markdown("---")
