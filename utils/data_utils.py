# utils/data_utils.py - Data Management Utilities
import os
import numpy as np
import tensorflow as tf
from PIL import Image
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class DataGenerator:
    """Custom data generator for fashion images"""
    
    def __init__(self, data_dir, batch_size=32, image_size=(224, 224), 
                 shuffle=True, augmentation=False):
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.augmentation = augmentation
        
        # Load class mapping
        self.class_names = self._get_class_names()
        self.num_classes = len(self.class_names)
        
        # Load file paths and labels
        self.file_paths, self.labels = self._load_data_paths()
        
        # Create TensorFlow dataset
        self.dataset = self._create_dataset()
    
    def _get_class_names(self):
        """Get class names from directory structure"""
        class_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        return sorted([d.name for d in class_dirs])
    
    def _load_data_paths(self):
        """Load all file paths and their corresponding labels"""
        file_paths = []
        labels = []
        
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = self.data_dir / class_name
            if class_dir.exists():
                for img_file in class_dir.glob('*'):
                    if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        file_paths.append(str(img_file))
                        labels.append(class_idx)
        
        return np.array(file_paths), np.array(labels)
    
    def _create_dataset(self):
        """Create TensorFlow dataset"""
        # Create dataset from file paths and labels
        dataset = tf.data.Dataset.from_tensor_slices((self.file_paths, self.labels))
        
        # Shuffle if required
        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=len(self.file_paths))
        
        # Map to load and preprocess images
        dataset = dataset.map(
            self._load_and_preprocess_image,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Batch and prefetch
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def _load_and_preprocess_image(self, file_path, label):
        """Load and preprocess a single image"""
        # Load image
        image = tf.io.read_file(file_path)
        image = tf.image.decode_image(image, channels=3)
        image = tf.cast(image, tf.float32)
        
        # Resize
        image = tf.image.resize(image, self.image_size)
        
        # Normalize
        image = image / 255.0
        
        # Apply augmentation if training
        if self.augmentation:
            image = self._apply_augmentation(image)
        
        # Convert label to one-hot
        label = tf.one_hot(label, self.num_classes)
        
        return image, label
    
    def _apply_augmentation(self, image):
        """Apply data augmentation"""
        # Random flip
        image = tf.image.random_flip_left_right(image)
        
        # Random rotation
        image = tf.image.rot90(image, k=tf.random.uniform([], 0, 4, dtype=tf.int32))
        
        # Random brightness and contrast
        image = tf.image.random_brightness(image, 0.1)
        image = tf.image.random_contrast(image, 0.8, 1.2)
        
        # Random saturation
        image = tf.image.random_saturation(image, 0.8, 1.2)
        
        # Ensure values are in valid range
        image = tf.clip_by_value(image, 0.0, 1.0)
        
        return image
    
    def __len__(self):
        """Return number of batches"""
        return len(self.file_paths) // self.batch_size
    
    def __iter__(self):
        """Make the generator iterable"""
        return iter(self.dataset)

def load_dataset(dataset_path):
    """Load dataset with train/validation/test splits"""
    dataset_path = Path(dataset_path)
    
    # Load splits
    train_gen = DataGenerator(
        dataset_path / 'train',
        batch_size=32,
        shuffle=True,
        augmentation=True
    )
    
    val_gen = DataGenerator(
        dataset_path / 'validation',
        batch_size=32,
        shuffle=False,
        augmentation=False
    )
    
    test_gen = DataGenerator(
        dataset_path / 'test',
        batch_size=32,
        shuffle=False,
        augmentation=False
    )
    
    return train_gen, val_gen, test_gen
