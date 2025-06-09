# scripts/train_model.py - Complete Training Pipeline
import os
import sys
import argparse
import tensorflow as tf
import numpy as np
from datetime import datetime
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_utils import DataGenerator, load_dataset
from utils.augmentation import create_augmentation_pipeline
from utils.metrics import calculate_comprehensive_metrics
from config.model_config import ModelConfig
from services.training_service import TrainingOrchestrator

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train Fashion AI Model')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset path')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--model-name', type=str, default='fashion_model', help='Model name')
    
    args = parser.parse_args()
    
    # Initialize training orchestrator
    orchestrator = TrainingOrchestrator()
    
    # Run training
    model_path = orchestrator.train_model(
        dataset_path=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        model_name=args.model_name
    )
    
    print(f"✅ Training completed! Model saved to: {model_path}")

if __name__ == '__main__':
    main()
