import tensorflow as tf
import numpy as np
from PIL import Image

def preprocess_image(image_path, target_size=(224, 224)):
    """
    Preprocess a single image for prediction.
    """
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_tumor_type(model, image_path, class_names):
    """
    Predict tumor type from an image.
    """
    img_array = preprocess_image(image_path)
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)
    return predicted_class, confidence