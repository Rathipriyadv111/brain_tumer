import tensorflow as tf
from tensorflow.keras.models import load_model
from scripts.data_preprocessing import load_and_preprocess_data
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

# Create output directory if it doesn't exist
os.makedirs('outputs', exist_ok=True)

# Load test data
test_generator = load_and_preprocess_data('data/test', target_size=(224, 224), batch_size=16, split='test')

# Load models
custom_cnn = load_model('models/custom_cnn.h5')
transfer_model = load_model('models/transfer_learning_model.h5')

# Evaluate custom CNN
predictions = custom_cnn.predict(test_generator)
y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes
class_names = list(test_generator.class_indices.keys())
print("Custom CNN Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# Evaluate transfer learning model
predictions = transfer_model.predict(test_generator)
y_pred_transfer = np.argmax(predictions, axis=1)
print("Transfer Learning Model Classification Report:")
print(classification_report(y_true, y_pred_transfer, target_names=class_names))

# Plot confusion matrix
cm = confusion_matrix(y_true, y_pred_transfer)  # Using transfer learning model
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix (Transfer Learning Model)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('outputs/confusion_matrix.png')
plt.close()

# Plot training history
with open('models/custom_cnn_history.pkl', 'rb') as f:
    custom_cnn_history = pickle.load(f)
with open('models/transfer_learning_model_history.pkl', 'rb') as f:
    transfer_model_history = pickle.load(f)

# Plot accuracy
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(custom_cnn_history['accuracy'], label='Custom CNN Train')
plt.plot(custom_cnn_history['val_accuracy'], label='Custom CNN Valid')
plt.plot(transfer_model_history['accuracy'], label='Transfer Model Train')
plt.plot(transfer_model_history['val_accuracy'], label='Transfer Model Valid')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(custom_cnn_history['loss'], label='Custom CNN Train')
plt.plot(custom_cnn_history['val_loss'], label='Custom CNN Valid')
plt.plot(transfer_model_history['loss'], label='Transfer Model Train')
plt.plot(transfer_model_history['val_loss'], label='Transfer Model Valid')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('outputs/training_history.png')
plt.close()