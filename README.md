Brain Tumor MRI Image Classification
Overview
This project develops a deep learning-based solution for classifying brain MRI images into multiple tumor types. It includes a custom CNN model and a transfer learning model using ResNet50, with a Streamlit web application for real-time tumor type predictions.
Project Structure
brain_tumor_classification/
│
├── data/                     # Dataset directory (not included in repo)
├── models/                   # Trained models
│   ├── custom_cnn.h5
│   └── transfer_learning_model.h5
├── scripts/                  # Python scripts
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   └── utils.py
├── app.py                    # Streamlit app
├── requirements.txt          # Dependencies
└── README.md                 # Project documentation

Setup Instructions

Clone the Repository:
git clone <repository-url>
cd brain_tumor_classification


Install Dependencies:
pip install -r requirements.txt


Dataset:

Place the Brain Tumor MRI Multi-Class Dataset in the data/ directory.
Ensure the dataset is organized with subfolders for each tumor type.


Run the Streamlit App:
streamlit run app.py



Usage

Training: Run model_training.py to train the custom CNN and transfer learning models.
Evaluation: Use model_evaluation.py to evaluate model performance and generate metrics/plots.
Deployment: Use app.py to launch the Streamlit app for real-time predictions.

Requirements

Python 3.8+
See requirements.txt for detailed dependencies.

Deliverables

Trained models (custom_cnn.h5, transfer_learning_model.h5)
Streamlit application (app.py)
Python scripts for preprocessing, training, and evaluation
Model comparison via evaluation metrics
Public GitHub repository with this README
