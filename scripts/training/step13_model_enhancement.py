# step13_model_enhancement.py - Take your working model to the next level

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import *
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import time

class ModelEnhancer:
    def __init__(self, base_model_path="working_fashion_model.keras"):
        self.base_model_path = base_model_path
        self.model = None
        self.dataset_path = "real_fashion_dataset_training"
        self.categories = [
            'shirts_blouses', 'tshirts_tops', 'dresses', 'pants_jeans',
            'shorts', 'skirts', 'jackets_coats', 'sweaters', 
            'shoes_sneakers', 'shoes_formal', 'bags_accessories'
        ]
        
    def load_working_model(self):
        """Load the working model as starting point"""
        print(f"📂 Loading working model: {self.base_model_path}")
        
        try:
            self.model = load_model(self.base_model_path)
            print("✅ Model loaded successfully!")
            
            # Get current performance metrics
            trainable_params = sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights])
            non_trainable_params = sum([tf.keras.backend.count_params(w) for w in self.model.non_trainable_weights])
            
            print(f"Model structure:")
            print(f"  Input shape: {self.model.input_shape}")
            print(f"  Output shape: {self.model.output_shape}")
            print(f"  Total parameters: {trainable_params + non_trainable_params:,}")
            print(f"  Trainable parameters: {trainable_params:,}")
            
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def setup_advanced_data_generators(self):
        """Setup more advanced data generators for enhancement"""
        print("📊 Setting up advanced data generators...")
        
        # Enhanced augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.15,
            zoom_range=0.2,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest'
        )
        
        val_test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create generators
        train_generator = train_datagen.flow_from_directory(
            os.path.join(self.dataset_path, 'train'),
            target_size=(224, 224),
            batch_size=16,
            class_mode='categorical',
            shuffle=True,
            classes=self.categories
        )
        
        val_generator = val_test_datagen.flow_from_directory(
            os.path.join(self.dataset_path, 'validation'),
            target_size=(224, 224),
            batch_size=16,
            class_mode='categorical',
            shuffle=False,
            classes=self.categories
        )
        
        test_generator = val_test_datagen.flow_from_directory(
            os.path.join(self.dataset_path, 'test'),
            target_size=(224, 224),
            batch_size=16,
            class_mode='categorical',
            shuffle=False,
            classes=self.categories
        )
        
        print(f"✅ Enhanced generators created:")
        print(f"   Train: {train_generator.samples} samples")
        print(f"   Validation: {val_generator.samples} samples")
        print(f"   Test: {test_generator.samples} samples")
        
        return train_generator, val_generator, test_generator
    
    def enhance_model(self, epochs=30):
        """Enhance the model with advanced training techniques"""
        print("🚀 Starting model enhancement...")
        
        # Setup data generators
        train_gen, val_gen, test_gen = self.setup_advanced_data_generators()
        
        # Unfreeze more MobileNetV2 layers for better feature learning
        # The base model is typically the second layer
        base_model = self.model.layers[1]
        
        # Unfreeze final layers of MobileNetV2
        for layer in base_model.layers[-30:]:  # Unfreeze last 30 layers
            layer.trainable = True
        
        # Custom learning rate scheduler
        def lr_scheduler(epoch, lr):
            if epoch < 5:
                return lr  # Keep initial rate for first 5 epochs
            elif epoch < 15:
                return lr * 0.8  # Reduce slightly
            else:
                return lr * 0.5  # Reduce more for final refinement
        
        # Advanced callbacks
        callbacks = [
            ModelCheckpoint(
                'enhanced_fashion_model.keras',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=4,
                min_lr=1e-6,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_accuracy',
                patience=8,
                restore_best_weights=True,
                verbose=1
            ),
            LearningRateScheduler(lr_scheduler),
            TensorBoard(
                log_dir=f'./logs/enhancement_{int(time.time())}',
                histogram_freq=1
            )
        ]
        
        # Recompile with very low learning rate for fine-tuning
        self.model.compile(
            optimizer=Adam(learning_rate=5e-5),  # Very low learning rate
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("📈 Starting advanced fine-tuning...")
        history = self.model.fit(
            train_gen,
            epochs=epochs,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate enhanced model
        print("\n📊 Evaluating enhanced model...")
        test_results = self.model.evaluate(test_gen, verbose=1)
        
        print(f"\n🎉 ENHANCEMENT COMPLETE!")
        print(f"Final Test Accuracy: {test_results[1]:.4f} ({test_results[1]*100:.1f}%)")
        print(f"Enhanced model saved: enhanced_fashion_model.keras")
        
        # Plot training progress
        self.plot_training_history(history)
        
        return history, test_results
    
    def plot_training_history(self, history):
        """Plot training history"""
        plt.figure(figsize=(12, 5))
        
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Val Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('model_enhancement_progress.png')
        plt.show()

def run_model_enhancement():
    """Run the model enhancement process"""
    
    print("🌟 MODEL ENHANCEMENT PIPELINE")
    print("=" * 50)
    
    enhancer = ModelEnhancer()
    
    # Step 1: Load working model
    if not enhancer.load_working_model():
        print("❌ Could not load working model")
        return False
    
    # Step 2: Enhance model
    history, results = enhancer.enhance_model(epochs=30)
    
    # Step 3: Report improvement
    print("\n🔍 ENHANCEMENT RESULTS:")
    print("=" * 30)
    print(f"✅ Model saved: enhanced_fashion_model.keras")
    print(f"✅ Final accuracy: {results[1]:.1%}")
    print(f"✅ Training graph saved: model_enhancement_progress.png")
    
    print("\n🚀 NEXT STEPS:")
    print("1. Update your app.py MODEL_PATH to 'enhanced_fashion_model.keras'")
    print("2. Restart your Flask app to use the enhanced model")
    
    return True

if __name__ == "__main__":
    run_model_enhancement()
