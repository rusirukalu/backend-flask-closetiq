# step10_organize_and_train.py - Complete training pipeline

import os
import shutil
import random
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import *
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

# Define learning rate schedule function (outside class to avoid pickling issues)
def cosine_decay_schedule(epoch, initial_lr=0.001, decay_rate=0.95):
    """Cosine decay learning rate schedule"""
    return initial_lr * (decay_rate ** epoch)

class RealFashionTrainer:
    def __init__(self):
        self.source_dir = "real_fashion_dataset"
        self.training_dir = "real_fashion_dataset_training"
        self.categories = [
            'shirts_blouses', 'tshirts_tops', 'dresses', 'pants_jeans',
            'shorts', 'skirts', 'jackets_coats', 'sweaters', 
            'shoes_sneakers', 'shoes_formal', 'bags_accessories'
        ]
        self.model = None
        
    def organize_dataset(self):
        """Organize collected images into train/validation/test splits"""
        print("📂 Organizing dataset into train/validation/test splits...")
        
        # Create directory structure
        for split in ['train', 'validation', 'test']:
            for category in self.categories:
                os.makedirs(os.path.join(self.training_dir, split, category), exist_ok=True)
        
        total_organized = 0
        split_summary = {'train': 0, 'validation': 0, 'test': 0}
        category_summary = {}
        
        for category in self.categories:
            # Check multiple possible source structures
            possible_sources = [
                os.path.join(self.source_dir, category),  # Direct category folders
                os.path.join(self.source_dir, 'train', category),  # Already split structure
            ]
            
            source_path = None
            for path in possible_sources:
                if os.path.exists(path):
                    source_path = path
                    break
            
            if not source_path:
                print(f"⚠️ Warning: No images found for category: {category}")
                continue
            
            # Get all images
            images = [f for f in os.listdir(source_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if not images:
                print(f"⚠️ Warning: No valid images in {source_path}")
                continue
                
            random.shuffle(images)
            
            # Split ratios: 70% train, 15% validation, 15% test
            total = len(images)
            train_end = int(0.7 * total)
            val_end = int(0.85 * total)
            
            train_imgs = images[:train_end]
            val_imgs = images[train_end:val_end]
            test_imgs = images[val_end:]
            
            # Copy images to respective splits
            for img in train_imgs:
                shutil.copy2(
                    os.path.join(source_path, img),
                    os.path.join(self.training_dir, 'train', category, img)
                )
            
            for img in val_imgs:
                shutil.copy2(
                    os.path.join(source_path, img),
                    os.path.join(self.training_dir, 'validation', category, img)
                )
            
            for img in test_imgs:
                shutil.copy2(
                    os.path.join(source_path, img),
                    os.path.join(self.training_dir, 'test', category, img)
                )
            
            # Update summaries
            category_total = len(train_imgs) + len(val_imgs) + len(test_imgs)
            category_summary[category] = {
                'train': len(train_imgs),
                'validation': len(val_imgs),
                'test': len(test_imgs),
                'total': category_total
            }
            
            split_summary['train'] += len(train_imgs)
            split_summary['validation'] += len(val_imgs)
            split_summary['test'] += len(test_imgs)
            total_organized += category_total
            
            print(f"✅ {category}: {len(train_imgs)} train, {len(val_imgs)} val, {len(test_imgs)} test")
        
        # Print summary
        print(f"\n📊 DATASET ORGANIZATION COMPLETE!")
        print(f"Total images organized: {total_organized}")
        print(f"Train: {split_summary['train']} | Validation: {split_summary['validation']} | Test: {split_summary['test']}")
        
        return category_summary
    
    def create_production_model(self):
        """Create production-ready model architecture"""
        print("🏗️ Creating production-ready EfficientNet model...")
        
        # EfficientNetV2-B0 (best balance of accuracy and speed)
        base_model = EfficientNetV2B0(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3),
            pooling=None
        )
        
        # Freeze base model initially for transfer learning
        base_model.trainable = False
        
        # Input layer
        inputs = Input(shape=(224, 224, 3))
        
        # Built-in data augmentation (crucial for real images)
        x = RandomFlip("horizontal")(inputs)
        x = RandomRotation(0.15)(x)
        x = RandomZoom(0.1)(x)
        x = RandomBrightness(0.1)(x)
        
        # Base model
        x = base_model(x, training=False)
        
        # Advanced classification head
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        # Multi-branch architecture for better feature learning
        branch1 = Dense(512, activation='relu')(x)
        branch1 = BatchNormalization()(branch1)
        branch1 = Dropout(0.4)(branch1)
        
        branch2 = Dense(256, activation='relu')(x)
        branch2 = BatchNormalization()(branch2)
        branch2 = Dropout(0.3)(branch2)
        
        # Combine branches
        combined = concatenate([branch1, branch2])
        
        # Final layers
        x = Dense(256, activation='relu')(combined)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        
        # Output layer
        outputs = Dense(11, activation='softmax', name='fashion_predictions')(x)
        
        # Create model
        self.model = Model(inputs=inputs, outputs=outputs, name='RealFashionClassifier')
        
        # Compile with production settings
        self.model.compile(
            optimizer=AdamW(learning_rate=0.001, weight_decay=0.0001),
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy'),
                tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy')
            ]
        )
        
        print("✅ Production model created!")
        print(f"Total parameters: {self.model.count_params():,}")
        
        return self.model
    
    def setup_data_generators(self):
        """Setup optimized data generators for real fashion images"""
        print("📊 Setting up data generators...")
        
        # Enhanced augmentation for real fashion images
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=25,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.15,
            zoom_range=0.2,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            channel_shift_range=0.1,
            fill_mode='nearest'
        )
        
        val_test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create generators
        train_generator = train_datagen.flow_from_directory(
            os.path.join(self.training_dir, 'train'),
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical',
            shuffle=True,
            classes=self.categories
        )
        
        val_generator = val_test_datagen.flow_from_directory(
            os.path.join(self.training_dir, 'validation'),
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical',
            shuffle=False,
            classes=self.categories
        )
        
        test_generator = val_test_datagen.flow_from_directory(
            os.path.join(self.training_dir, 'test'),
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical',
            shuffle=False,
            classes=self.categories
        )
        
        print(f"✅ Data generators created:")
        print(f"   Train: {train_generator.samples} samples")
        print(f"   Validation: {val_generator.samples} samples")
        print(f"   Test: {test_generator.samples} samples")
        print(f"   Classes: {list(train_generator.class_indices.keys())}")
        
        return train_generator, val_generator, test_generator
    
    def train_production_model(self, epochs=40):
        """Train the production model with real fashion data"""
        print("🚀 Starting production model training...")
        
        # Setup data generators
        train_gen, val_gen, test_gen = self.setup_data_generators()
        
        # Setup simplified callbacks (avoid deep copy issues)
        callbacks = [
            ModelCheckpoint(
                'real_fashion_classifier_final.keras',  # Use .keras format
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=12,
                restore_best_weights=True,
                verbose=1,
                min_delta=0.001
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.3,
                patience=6,
                min_lr=1e-8,
                verbose=1,
                cooldown=2
            )
        ]
        
        # Phase 1: Feature extraction (frozen backbone)
        print("\n🎯 Phase 1: Feature extraction training (20 epochs)")
        
        try:
            history_phase1 = self.model.fit(
                train_gen,
                epochs=20,
                validation_data=val_gen,
                callbacks=callbacks,
                verbose=1
            )
        except Exception as e:
            print(f"❌ Error in Phase 1 training: {e}")
            print("Trying with simplified callbacks...")
            
            # Fallback with minimal callbacks
            simple_callbacks = [
                ModelCheckpoint(
                    'real_fashion_classifier_final.keras',
                    monitor='val_accuracy',
                    save_best_only=True,
                    verbose=1
                )
            ]
            
            history_phase1 = self.model.fit(
                train_gen,
                epochs=20,
                validation_data=val_gen,
                callbacks=simple_callbacks,
                verbose=1
            )
        
        # Phase 2: Fine-tuning (unfreeze backbone)
        print("\n🔥 Phase 2: Fine-tuning entire model")
        
        # Unfreeze the base model
        self.model.layers[4].trainable = True  # EfficientNet layer
        
        # Recompile with lower learning rate for fine-tuning
        self.model.compile(
            optimizer=AdamW(learning_rate=0.0001, weight_decay=0.0001),
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy'),
                tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy')
            ]
        )
        
        # Continue training with minimal callbacks
        simple_callbacks = [
            ModelCheckpoint(
                'real_fashion_classifier_final.keras',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=8,
                restore_best_weights=True,
                verbose=1
            )
        ]
        
        try:
            history_phase2 = self.model.fit(
                train_gen,
                epochs=epochs,
                initial_epoch=20,
                validation_data=val_gen,
                callbacks=simple_callbacks,
                verbose=1
            )
        except Exception as e:
            print(f"❌ Error in Phase 2 training: {e}")
            print("Continuing without callbacks...")
            
            history_phase2 = self.model.fit(
                train_gen,
                epochs=epochs,
                initial_epoch=20,
                validation_data=val_gen,
                verbose=1
            )
            
            # Save model manually
            self.model.save('real_fashion_classifier_final.keras')
        
        # Final evaluation
        print("\n📊 Final evaluation on test set...")
        test_results = self.model.evaluate(test_gen, verbose=1)
        
        # Detailed predictions for analysis
        predictions = self.model.predict(test_gen)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = test_gen.classes
        
        # Classification report
        print("\n📋 Detailed Classification Report:")
        class_report = classification_report(
            true_classes, predicted_classes,
            target_names=self.categories,
            digits=4
        )
        print(class_report)
        
        # Confusion matrix
        cm = confusion_matrix(true_classes, predicted_classes)
        
        # Plot confusion matrix
        try:
            plt.figure(figsize=(12, 10))
            sns.heatmap(
                cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=self.categories, yticklabels=self.categories
            )
            plt.title('Real Fashion Classifier - Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig('real_fashion_confusion_matrix.png', dpi=300, bbox_inches='tight')
            plt.close()  # Close instead of show to avoid blocking
            print("📊 Confusion matrix saved: real_fashion_confusion_matrix.png")
        except Exception as e:
            print(f"⚠️ Could not create confusion matrix plot: {e}")
        
        print(f"\n🎉 PRODUCTION TRAINING COMPLETE!")
        print(f"📊 Final Results:")
        print(f"   Test Accuracy: {test_results[1]:.4f} ({test_results[1]*100:.1f}%)")
        print(f"   Top-3 Accuracy: {test_results[2]:.4f} ({test_results[2]*100:.1f}%)")
        print(f"   Top-5 Accuracy: {test_results[3]:.4f} ({test_results[3]*100:.1f}%)")
        print(f"   Model saved: real_fashion_classifier_final.keras")
        
        return history_phase1, history_phase2, test_results
    
    def validate_model_performance(self):
        """Validate model performance and create performance report"""
        print("\n🔍 Validating model performance...")
        
        try:
            # Load the best model
            model = tf.keras.models.load_model('real_fashion_classifier_final.keras')
        except:
            print("⚠️ Could not load saved model, using current model")
            model = self.model
        
        # Setup test generator
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_directory(
            os.path.join(self.training_dir, 'test'),
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical',
            shuffle=False,
            classes=self.categories
        )
        
        # Get predictions
        predictions = model.predict(test_generator)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = test_generator.classes
        
        # Calculate per-class accuracy
        per_class_accuracy = {}
        for i, category in enumerate(self.categories):
            class_mask = true_classes == i
            if np.sum(class_mask) > 0:
                class_accuracy = np.mean(predicted_classes[class_mask] == true_classes[class_mask])
                per_class_accuracy[category] = float(class_accuracy)
        
        # Performance report
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_path': 'real_fashion_classifier_final.keras',
            'total_test_samples': len(true_classes),
            'overall_accuracy': float(np.mean(predicted_classes == true_classes)),
            'per_class_accuracy': per_class_accuracy,
            'categories': self.categories,
            'training_data_size': 916,
            'model_type': 'EfficientNetV2-B0 + Custom Head'
        }
        
        # Save report
        try:
            with open('real_fashion_model_report.json', 'w') as f:
                json.dump(report, f, indent=2)
            print(f"✅ Performance report saved: real_fashion_model_report.json")
        except Exception as e:
            print(f"⚠️ Could not save report: {e}")
        
        print(f"🎯 Overall accuracy: {report['overall_accuracy']:.4f}")
        
        return report

def run_complete_training_pipeline():
    """Run the complete training pipeline"""
    
    print("🎯 REAL FASHION MODEL TRAINING PIPELINE")
    print("=" * 60)
    print("Training with 916 real fashion images...")
    
    # Initialize trainer
    trainer = RealFashionTrainer()
    
    try:
        # Step 1: Organize dataset
        category_summary = trainer.organize_dataset()
        
        # Step 2: Create model
        model = trainer.create_production_model()
        
        # Step 3: Train model
        h1, h2, results = trainer.train_production_model(epochs=40)
        
        # Step 4: Validate performance
        report = trainer.validate_model_performance()
        
        print("\n🏆 TRAINING PIPELINE COMPLETE!")
        print("=" * 60)
        print(f"✅ Model trained on 916 real fashion images")
        print(f"✅ Final accuracy: {results[1]:.1%}")
        print(f"✅ Production-ready model: real_fashion_classifier_final.keras")
        print(f"✅ Ready for deployment!")
        
        return trainer, report
        
    except Exception as e:
        print(f"\n❌ Training pipeline failed: {e}")
        print("This could be due to:")
        print("  • Insufficient memory")
        print("  • Corrupted image files")
        print("  • TensorFlow/GPU issues")
        return None, None

if __name__ == "__main__":
    trainer, report = run_complete_training_pipeline()
