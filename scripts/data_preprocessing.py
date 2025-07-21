import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_and_preprocess_data(data_dir, target_size=(224, 224), batch_size=16, split='train'):
    """
    Load and preprocess brain MRI images from the dataset directory.
    Args:
        data_dir (str): Path to the dataset directory (e.g., 'data/train', 'data/test', 'data/valid').
        target_size (tuple): Image size (height, width).
        batch_size (int): Number of images per batch.
        split (str): Dataset split ('train', 'test', or 'valid').
    Returns:
        Generator for the specified dataset split.
    """
    if split == 'train':
        datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            zoom_range=0.2,
            brightness_range=[0.8, 1.2]
        )
    else:  # 'test' or 'valid'
        datagen = ImageDataGenerator(rescale=1./255)
    
    generator = datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=(split == 'train')  # Shuffle for training, not for test/valid
    )
    
    return generator

def check_class_distribution(data_dir):
    """
    Check class distribution in the dataset.
    Args:
        data_dir (str): Path to the dataset directory (e.g., 'data/train').
    Returns:
        dict: Number of images per class.
    """
    class_counts = {}
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):  # Only include directories
            class_counts[class_name] = len(os.listdir(class_path))
    return class_counts