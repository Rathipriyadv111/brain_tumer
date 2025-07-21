import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D
from data_preprocessing import load_and_preprocess_data  # Changed from scripts.data_preprocessing
import os

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Load training and validation data
train_generator = load_and_preprocess_data('data/train', target_size=(224, 224), batch_size=16, split='train')
valid_generator = load_and_preprocess_data('data/valid', target_size=(224, 224), batch_size=16, split='valid')

# Define custom CNN model
custom_cnn = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')  # 4 classes: glioma, meningioma, no_tumor, pituitary
])

# Compile custom CNN
custom_cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train custom CNN
custom_cnn_history = custom_cnn.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=10,
    verbose=1
)

# Save custom CNN
custom_cnn.save('models/custom_cnn.h5')

# Define transfer learning model (ResNet50)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze base model layers

transfer_model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')  # 4 classes
])

# Compile transfer learning model
transfer_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train transfer learning model
transfer_model_history = transfer_model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=10,
    verbose=1
)

# Save transfer learning model
transfer_model.save('models/transfer_learning_model.h5')

# Save training history for plotting in evaluation
import pickle
with open('models/custom_cnn_history.pkl', 'wb') as f:
    pickle.dump(custom_cnn_history.history, f)
with open('models/transfer_learning_model_history.pkl', 'wb') as f:
    pickle.dump(transfer_model_history.history, f)