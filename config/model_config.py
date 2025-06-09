# config/model_config.py - Model-specific Configuration
import os
from pathlib import Path

class ModelConfig:
    """Model training and architecture configuration"""
    
    # Model architecture
    BASE_MODEL = 'EfficientNetV2B0'
    INPUT_SHAPE = (224, 224, 3)
    DROPOUT_RATE = 0.3
    DENSE_UNITS = 512
    
    # Training parameters
    DEFAULT_EPOCHS = 50
    DEFAULT_BATCH_SIZE = 32
    DEFAULT_LEARNING_RATE = 0.001
    FINE_TUNE_EPOCHS = 20
    FINE_TUNE_LEARNING_RATE = 0.0001
    
    # Data augmentation
    ROTATION_RANGE = 20
    WIDTH_SHIFT_RANGE = 0.2
    HEIGHT_SHIFT_RANGE = 0.2
    BRIGHTNESS_RANGE = [0.8, 1.2]
    ZOOM_RANGE = 0.1
    
    # Class names (from your previous work)
    CLASS_NAMES = [
        'shirts_blouses', 'tshirts_tops', 'dresses', 'pants_jeans',
        'shorts', 'skirts', 'jackets_coats', 'sweaters', 
        'shoes_sneakers', 'shoes_formal', 'bags_accessories'
    ]
    
    # Model paths
    MODELS_DIR = Path('models/trained')
    CHECKPOINTS_DIR = Path('models/checkpoints')
    EXPORTS_DIR = Path('models/exports')
    
    # Performance thresholds
    MIN_ACCURACY = 0.85
    MIN_VALIDATION_ACCURACY = 0.80
    
    # Training callbacks
    EARLY_STOPPING_PATIENCE = 10
    REDUCE_LR_PATIENCE = 5
    REDUCE_LR_FACTOR = 0.5
    MIN_LR = 1e-7
