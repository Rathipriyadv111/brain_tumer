🧠 Brain Tumor MRI Image Classification
This project presents a deep learning-based solution to classify brain MRI images into multiple tumor types using both a Custom CNN and a Transfer Learning approach with ResNet50. A user-friendly Streamlit web app is included for real-time predictions.

📂 Project Structure
bash
Copy
Edit
brain_tumor_classification/
│
├── data/                       # Dataset directory (not included in repo)
├── models/                     # Trained models
│   ├── custom_cnn.h5
│   └── transfer_learning_model.h5
├── scripts/                    # Python scripts
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   └── utils.py
├── app.py                      # Streamlit app
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
⚙️ Setup Instructions
1. Clone the Repository
bash
Copy
Edit
git clone https://github.com/<your-username>/brain_tumor_classification.git
cd brain_tumor_classification
2. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
3. Add Dataset
Place the Brain Tumor MRI Multi-Class Dataset inside the data/ directory. Ensure the dataset follows the format:

kotlin
Copy
Edit
data/
├── glioma/
├── meningioma/
├── pituitary/
└── no_tumor/
4. Run the Streamlit App
bash
Copy
Edit
streamlit run app.py
🚀 Usage
Training
Run the following to train both models:

bash
Copy
Edit
python scripts/model_training.py
Evaluation
Generate evaluation metrics and visualizations:

bash
Copy
Edit
python scripts/model_evaluation.py
Deployment
Launch the web app:

bash
Copy
Edit
streamlit run app.py
🧾 Requirements
Python 3.8+

TensorFlow

Streamlit

NumPy

Matplotlib

Seaborn

Scikit-learn

Pillow

Altair

📄 All dependencies are listed in requirements.txt.

📦 Deliverables
✅ Trained models: custom_cnn.h5, transfer_learning_model.h5

✅ Streamlit application: app.py

✅ Complete pipeline scripts: Preprocessing, training, evaluation

✅ Model comparison through evaluation metrics

✅ Public GitHub repository with documentation

👩‍💻 Author
Rathi Priya
Brain Tumor Classification using Deep Learning | 2025
