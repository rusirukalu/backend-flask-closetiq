# services/training_service.py - Training Service Integration
import os
import sys
import tensorflow as tf
import numpy as np
from datetime import datetime
import json
import logging
from pathlib import Path

# Import your existing step modules (after migration)
sys.path.append('scripts/training')

logger = logging.getLogger(__name__)

class TrainingOrchestrator:
    """Orchestrates training using your existing step modules"""
    
    def __init__(self):
        self.models_dir = Path('models/trained')
        self.checkpoints_dir = Path('models/checkpoints')
        self.metadata_dir = Path('models/metadata')
        
        # Create directories
        for directory in [self.models_dir, self.checkpoints_dir, self.metadata_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def train_new_model(self, config=None):
        """Train a new model using your existing training pipeline"""
        try:
            logger.info("🚀 Starting model training...")
            
            from step1_advanced_attribute_classifier import main as step1_main
            from step2_data_improvement import main as step2_main
            from step3_advanced_retraining import main as step3_main
            
            # Run your training pipeline
            step1_result = step1_main()
            step2_result = step2_main()
            step3_result = step3_main()
            
            # Get the final model path
            model_path = self._get_latest_model_path()
            
            # Save training metadata
            metadata = {
                'training_date': datetime.now().isoformat(),
                'model_path': str(model_path),
                'training_steps': ['step1', 'step2', 'step3'],
                'success': True
            }
            
            metadata_path = self.metadata_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"✅ Training completed: {model_path}")
            return str(model_path)
            
        except Exception as e:
            logger.error(f"❌ Training failed: {str(e)}")
            raise
    
    def retrain_model(self, base_model_path, new_data_path):
        """Retrain existing model with new data"""
        try:
            logger.info("🔄 Starting model retraining...")
            
            # Import your retraining module
            from step3_advanced_retraining import retrain_with_new_data
            
            # Run retraining
            new_model_path = retrain_with_new_data(base_model_path, new_data_path)
            
            logger.info(f"✅ Retraining completed: {new_model_path}")
            return new_model_path
            
        except Exception as e:
            logger.error(f"❌ Retraining failed: {str(e)}")
            raise
    
    def _get_latest_model_path(self):
        """Get the latest trained model"""
        model_files = list(self.models_dir.glob("*.keras"))
        if model_files:
            # Sort by modification time and return the latest
            latest_model = max(model_files, key=os.path.getmtime)
            return latest_model
        else:
            return self.models_dir / "final_enhanced_model.keras"
