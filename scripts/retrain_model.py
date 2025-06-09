# scripts/retrain_model.py - Model Retraining Pipeline
import os
import sys
import argparse
import tensorflow as tf
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.training_service import TrainingOrchestrator

def main():
    """Main retraining function"""
    parser = argparse.ArgumentParser(description='Retrain Fashion AI Model')
    parser.add_argument('--base-model', type=str, required=True, help='Base model path')
    parser.add_argument('--new-data', type=str, required=True, help='New dataset path')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--freeze-layers', type=int, default=15, help='Layers to freeze')
    
    args = parser.parse_args()
    
    # Initialize training orchestrator
    orchestrator = TrainingOrchestrator()
    
    # Run retraining
    model_path = orchestrator.retrain_model(
        base_model_path=args.base_model,
        new_data_path=args.new_data,
        epochs=args.epochs,
        freeze_layers=args.freeze_layers
    )
    
    print(f"✅ Retraining completed! Model saved to: {model_path}")

if __name__ == '__main__':
    main()
