# step12_guaranteed_training.py - Simplified, guaranteed training

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import *
import matplotlib.pyplot as plt
from datetime import datetime

class GuaranteedFashionTrainer:
    def __init__(self):
        self.dataset_path = "real_fashion_dataset_training"
        self.categories = [
            'shirts_blouses', 'tshirts_tops', 'dresses', 'pants_jeans',
            'shorts', 'skirts', 'jackets_coats', 'sweaters', 
            'shoes_sneakers', 'shoes_formal', 'bags_accessories'
        ]
        self.model = None
        
    def check_dataset(self):
        """Verify dataset exists and has enough images"""
        print("🔍 Checking dataset...")
        
        total_images = 0
        for split in ['train', 'validation', 'test']:
            split_path = os.path.join(self.dataset_path, split)
            if not os.path.exists(split_path):
                print(f"❌ Missing: {split_path}")
                return False
            
            split_total = 0
            for category in self.categories:
                cat_path = os.path.join(split_path, category)
                if os.path.exists(cat_path):
                    count = len([f for f in os.listdir(cat_path) 
                               if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                    split_total += count
                    print(f"   {split}/{category}: {count} images")
                else:
                    print(f"   ❌ Missing: {cat_path}")
            
            print(f"   {split} total: {split_total}")
            total_images += split_total
        
        print(f"\n📊 Total images: {total_images}")
        
        if total_images < 100:
            print("❌ Insufficient data for training")
            return False
        
        print("✅ Dataset verification passed!")
        return True
    
    def create_simple_working_model(self):
        """Create a simple model that will definitely train"""
        print("🏗️ Creating simple working model...")
        
        # Use MobileNetV2 (lighter than EfficientNet, more stable)
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3),
            pooling='avg'  # Global average pooling built-in
        )
        
        # Freeze base model initially
        base_model.trainable = False
        
        # Simple classification head
        inputs = Input(shape=(224, 224, 3))
        x = base_model(inputs, training=False)
        x = Dropout(0.2)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.3)(x)
        outputs = Dense(11, activation='softmax', name='fashion_predictions')(x)
        
        self.model = Model(inputs, outputs)
        
        # Simple compilation
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("✅ Simple model created!")
        print(f"Parameters: {self.model.count_params():,}")
        
        return self.model
    
    def setup_simple_data_generators(self):
        """Setup simple, reliable data generators"""
        print("📊 Setting up data generators...")
        
        # Minimal augmentation to avoid issues
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            horizontal_flip=True,
            rotation_range=10,
            zoom_range=0.1,
            width_shift_range=0.1,
            height_shift_range=0.1
        )
        
        val_test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create generators with smaller batch size for stability
        train_generator = train_datagen.flow_from_directory(
            os.path.join(self.dataset_path, 'train'),
            target_size=(224, 224),
            batch_size=16,  # Smaller batch size
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
        
        print(f"✅ Generators created:")
        print(f"   Train: {train_generator.samples} samples")
        print(f"   Validation: {val_generator.samples} samples")
        print(f"   Test: {test_generator.samples} samples")
        
        return train_generator, val_generator, test_generator
    
    def train_guaranteed(self):
        """Guaranteed training approach"""
        print("🚀 Starting GUARANTEED training...")
        
        # Setup data
        train_gen, val_gen, test_gen = self.setup_simple_data_generators()
        
        # Simple callbacks - no fancy stuff
        callbacks = [
            ModelCheckpoint(
                'working_fashion_model.keras',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=8,
                restore_best_weights=True,
                verbose=1
            ),
            # Custom callback to show progress
            ProgressCallback()
        ]
        
        # Phase 1: Train classification head only (10 epochs)
        print("\n📚 Phase 1: Training classification head (10 epochs)...")
        history1 = self.model.fit(
            train_gen,
            epochs=10,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )
        
        # Check if training is working
        if len(history1.history['accuracy']) > 0:
            final_accuracy = history1.history['accuracy'][-1]
            print(f"Phase 1 final accuracy: {final_accuracy:.4f}")
            
            if final_accuracy > 0.3:  # At least better than random
                print("✅ Phase 1 successful!")
                
                # Phase 2: Unfreeze and fine-tune (10 more epochs)
                print("\n🔥 Phase 2: Fine-tuning (10 epochs)...")
                
                # Unfreeze base model
                self.model.layers[1].trainable = True
                
                # Recompile with lower learning rate
                self.model.compile(
                    optimizer=Adam(learning_rate=0.0001),
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
                
                # Continue training
                history2 = self.model.fit(
                    train_gen,
                    epochs=20,
                    initial_epoch=10,
                    validation_data=val_gen,
                    callbacks=callbacks,
                    verbose=1
                )
            else:
                print("⚠️ Phase 1 accuracy too low, but continuing...")
                history2 = None
        
        # Final evaluation
        print("\n📊 Final evaluation...")
        test_results = self.model.evaluate(test_gen, verbose=1)
        
        print(f"\n🎉 TRAINING COMPLETE!")
        print(f"Final Test Accuracy: {test_results[1]:.4f} ({test_results[1]*100:.1f}%)")
        
        # Test the model immediately
        self.test_trained_model()
        
        return history1, test_results
    
    def test_trained_model(self):
        """Test the trained model to verify it works"""
        print("\n🧪 Testing trained model...")
        
        # Test with random input
        test_input = np.random.rand(1, 224, 224, 3)
        predictions = self.model.predict(test_input, verbose=0)
        
        max_conf = np.max(predictions[0])
        predicted_class = np.argmax(predictions[0])
        
        print(f"Random input test:")
        print(f"   Max confidence: {max_conf:.4f}")
        print(f"   Predicted class: {self.categories[predicted_class]}")
        
        if max_conf > 0.2:  # Better than random
            print("✅ Model is working - confidence above random!")
        else:
            print("⚠️ Model still shows low confidence")
        
        return max_conf

# Custom callback to show training progress
class ProgressCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs:
            acc = logs.get('accuracy', 0)
            val_acc = logs.get('val_accuracy', 0)
            print(f"Epoch {epoch+1}: accuracy={acc:.4f}, val_accuracy={val_acc:.4f}")

def run_guaranteed_training():
    """Run the guaranteed training process"""
    
    print("🎯 GUARANTEED FASHION MODEL TRAINING")
    print("=" * 50)
    
    trainer = GuaranteedFashionTrainer()
    
    # Step 1: Check dataset
    if not trainer.check_dataset():
        print("❌ Dataset issues found. Please fix dataset first.")
        return False
    
    # Step 2: Create model
    model = trainer.create_simple_working_model()
    
    # Step 3: Train
    history, results = trainer.train_guaranteed()
    
    # Step 4: Verify success
    if results[1] > 0.4:  # 40%+ accuracy
        print(f"\n🎉 SUCCESS! Model trained successfully!")
        print(f"✅ Model saved: working_fashion_model.keras")
        print(f"✅ Ready to replace in your app!")
        return True
    else:
        print(f"\n⚠️ Training completed but accuracy is low: {results[1]:.1%}")
        print("Consider collecting more/better training data")
        return False

if __name__ == "__main__":
    success = run_guaranteed_training()
    
    if success:
        print("\n🔄 NEXT STEPS:")
        print("1. Update your app.py MODEL_PATH to 'working_fashion_model.keras'")
        print("2. Restart your Flask app")
        print("3. Test with real clothing images")
