# step11_debug_model.py - Comprehensive Model Debugging

import os
import numpy as np
import tensorflow as tf
from PIL import Image
import json

class ModelDebugger:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        
    def load_and_inspect_model(self):
        """Load model and inspect its properties"""
        print("🔍 DEBUGGING MODEL ISSUES")
        print("=" * 50)
        
        # Check if model file exists
        if not os.path.exists(self.model_path):
            print(f"❌ Model file not found: {self.model_path}")
            print("Available model files:")
            for file in os.listdir('.'):
                if file.endswith(('.h5', '.keras')):
                    print(f"  - {file}")
            return False
        
        try:
            # Load model
            print(f"📁 Loading model: {self.model_path}")
            self.model = tf.keras.models.load_model(self.model_path)
            print("✅ Model loaded successfully!")
            
            # Inspect model architecture
            print(f"\n🏗️ MODEL ARCHITECTURE:")
            print(f"Input shape: {self.model.input_shape}")
            print(f"Output shape: {self.model.output_shape}")
            print(f"Total parameters: {self.model.count_params():,}")
            
            # Check output layer
            output_layer = self.model.layers[-1]
            print(f"Output layer: {output_layer.name}")
            print(f"Output units: {output_layer.units if hasattr(output_layer, 'units') else 'N/A'}")
            print(f"Activation: {output_layer.activation.__name__ if hasattr(output_layer, 'activation') else 'N/A'}")
            
            # Model summary (first few and last few layers)
            print(f"\n📋 MODEL SUMMARY:")
            print("First 5 layers:")
            for i, layer in enumerate(self.model.layers[:5]):
                print(f"  {i}: {layer.name} ({layer.__class__.__name__})")
            
            print("...")
            print("Last 5 layers:")
            for i, layer in enumerate(self.model.layers[-5:], len(self.model.layers)-5):
                print(f"  {i}: {layer.name} ({layer.__class__.__name__})")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def test_model_predictions(self):
        """Test model with various inputs"""
        if not self.model:
            print("❌ Model not loaded")
            return
        
        print(f"\n🧪 TESTING MODEL PREDICTIONS:")
        
        # Test 1: Random noise
        print("1. Testing with random noise...")
        random_input = np.random.rand(1, 224, 224, 3)
        try:
            predictions = self.model.predict(random_input, verbose=0)
            print(f"   Prediction shape: {predictions.shape}")
            print(f"   Raw predictions: {predictions[0]}")
            print(f"   Max confidence: {np.max(predictions[0]):.4f}")
            print(f"   Min confidence: {np.min(predictions[0]):.4f}")
            print(f"   Sum of predictions: {np.sum(predictions[0]):.4f}")
            
            # Check if predictions sum to 1 (proper softmax)
            if abs(np.sum(predictions[0]) - 1.0) > 0.01:
                print("   ⚠️ WARNING: Predictions don't sum to 1 - model might have issues")
            
        except Exception as e:
            print(f"   ❌ Prediction failed: {e}")
            return
        
        # Test 2: Check if model is undertrained
        print(f"\n2. Checking if model is undertrained...")
        predictions_list = []
        for i in range(5):
            test_input = np.random.rand(1, 224, 224, 3)
            pred = self.model.predict(test_input, verbose=0)
            predictions_list.append(np.max(pred[0]))
        
        avg_confidence = np.mean(predictions_list)
        print(f"   Average max confidence over 5 random inputs: {avg_confidence:.4f}")
        
        if avg_confidence < 0.2:
            print("   ❌ ISSUE: Model shows very low confidence - likely undertrained")
        elif avg_confidence > 0.9:
            print("   ⚠️ WARNING: Model shows very high confidence on random noise - might be overfitted")
        else:
            print("   ✅ Model confidence levels seem reasonable")
    
    def check_category_mapping(self):
        """Check if category mapping matches training"""
        print(f"\n🏷️ CHECKING CATEGORY MAPPING:")
        
        # Expected categories from your training
        expected_categories = [
            'shirts_blouses', 'tshirts_tops', 'dresses', 'pants_jeans',
            'shorts', 'skirts', 'jackets_coats', 'sweaters', 
            'shoes_sneakers', 'shoes_formal', 'bags_accessories'
        ]
        
        model_output_size = self.model.output_shape[-1]
        print(f"Model output size: {model_output_size}")
        print(f"Expected categories: {len(expected_categories)}")
        
        if model_output_size != len(expected_categories):
            print(f"❌ MISMATCH: Model expects {model_output_size} categories but app has {len(expected_categories)}")
            print("This could cause wrong classifications!")
        else:
            print("✅ Category count matches")
        
        print(f"\nExpected category order:")
        for i, cat in enumerate(expected_categories):
            print(f"  {i}: {cat}")
    
    def test_with_actual_image(self, image_path=None):
        """Test with an actual image if available"""
        print(f"\n📸 TESTING WITH ACTUAL IMAGE:")
        
        if not image_path:
            # Look for any image in current directory
            image_files = [f for f in os.listdir('.') if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if not image_files:
                print("No image files found for testing")
                return
            image_path = image_files[0]
            print(f"Using: {image_path}")
        
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image = image.resize((224, 224), Image.LANCZOS)
            image_array = np.array(image) / 255.0
            image_array = np.expand_dims(image_array, axis=0)
            
            # Make prediction
            predictions = self.model.predict(image_array, verbose=0)
            predicted_idx = np.argmax(predictions[0])
            confidence = predictions[0][predicted_idx]
            
            print(f"   Predicted class index: {predicted_idx}")
            print(f"   Confidence: {confidence:.4f}")
            print(f"   All predictions: {predictions[0]}")
            
        except Exception as e:
            print(f"   ❌ Error testing with image: {e}")

def run_model_debugging():
    """Run complete model debugging"""
    
    # Try different model paths
    possible_paths = [
        'real_fashion_classifier_final.keras',
        'real_fashion_classifier_final.h5',
        'emergency_baseline_model.h5',
        'final_working_model.h5'
    ]
    
    model_path = None
    for path in possible_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if not model_path:
        print("❌ No model file found! Available files:")
        for file in os.listdir('.'):
            if file.endswith(('.h5', '.keras')):
                print(f"  - {file}")
        return
    
    print(f"🎯 Debugging model: {model_path}")
    
    debugger = ModelDebugger(model_path)
    
    # Run debugging steps
    if debugger.load_and_inspect_model():
        debugger.test_model_predictions()
        debugger.check_category_mapping()
        debugger.test_with_actual_image()
    
    print(f"\n💡 RECOMMENDATIONS:")
    print("1. Check if training completed successfully")
    print("2. Verify category mapping matches training data")
    print("3. Ensure model was saved after training")
    print("4. Consider retraining if confidence is consistently low")

if __name__ == "__main__":
    run_model_debugging()
