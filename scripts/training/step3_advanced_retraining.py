# step3_advanced_retraining.py - Advanced Model Retraining

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2B0, MobileNetV3Large
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import AdamW, Adam
from tensorflow.keras.callbacks import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.metrics import TopKCategoricalAccuracy
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import json
from datetime import datetime
from tqdm import tqdm

# Define learning rate schedule function outside class to avoid serialization issues
def cosine_decay_schedule(epoch):
    """Learning rate schedule function"""
    return 0.001 * 0.95 ** epoch

class AdvancedModelTrainer:
    def __init__(self, dataset_path="improved_fashion_6cat", model_save_path="superior_fashion_classifier.keras"):
        self.dataset_path = dataset_path
        self.model_save_path = model_save_path  # Changed to .keras format
        self.categories = ['bottom', 'dress', 'formal', 'outerwear', 'shoes', 'top']
        self.num_classes = len(self.categories)
        
        # Based on Step 1 confusion analysis - these pairs need special attention
        self.confusion_pairs = [
            ('dress', 'top'),      # 100% confusion rate
            ('outerwear', 'top'),  # 100% confusion rate  
            ('shoes', 'formal'),   # 70% confusion rate
            ('formal', 'dress'),   # 50% confusion rate
        ]
        
        self.training_history = {}
        self.model = None
        
    def create_enhanced_model(self, architecture='efficientnet'):
        """Create enhanced model architecture with attention mechanisms"""
        print(f"🏗️ Building enhanced {architecture} model...")
        
        # Input layer
        inputs = Input(shape=(224, 224, 3))
        
        # Data augmentation layer (built into model)
        x = RandomFlip("horizontal")(inputs)
        x = RandomRotation(0.1)(x)
        x = RandomZoom(0.1)(x)
        x = RandomBrightness(0.1)(x)
        
        # Base model selection
        if architecture == 'efficientnet':
            base_model = EfficientNetV2B0(
                include_top=False,
                weights='imagenet',
                input_tensor=x,
                pooling=None
            )
        else:  # mobilenet fallback
            base_model = MobileNetV3Large(
                include_top=False,
                weights='imagenet',
                input_tensor=x,
                pooling=None
            )
        
        # Freeze base model initially
        base_model.trainable = False
        
        # Enhanced classification head with attention
        x = base_model.output
        
        # Global Average Pooling
        x = GlobalAveragePooling2D()(x)
        
        # Get the feature dimension dynamically
        feature_dim = x.shape[-1]  # Will be 1280 for EfficientNetV2B0
        
        # Attention mechanism for confusion-focused learning - FIXED
        attention = Dense(feature_dim // 4, activation='relu', name='attention_layer')(x)
        attention = Dense(feature_dim, activation='sigmoid', name='attention_weights')(attention)
        x_attended = multiply([x, attention], name='attention_applied')
        
        # Multi-scale feature processing
        # Branch 1: Direct features
        branch1 = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(x_attended)
        branch1 = BatchNormalization()(branch1)
        branch1 = Dropout(0.4)(branch1)
        
        # Branch 2: Reduced features for fine-grained distinctions
        branch2 = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(x_attended)
        branch2 = BatchNormalization()(branch2)
        branch2 = Dropout(0.3)(branch2)
        
        # Combine branches
        combined = concatenate([branch1, branch2])
        
        # Final classification layers
        x = Dense(384, activation='relu', kernel_regularizer=l2(0.001))(combined)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        
        x = Dense(192, activation='relu', kernel_regularizer=l2(0.001))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        
        # Output layer
        outputs = Dense(self.num_classes, activation='softmax', name='predictions')(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs, name=f'enhanced_{architecture}_classifier')
        
        print(f"✅ Enhanced model created with {model.count_params():,} parameters")
        print(f"   Feature dimension: {feature_dim}")
        
        return model
    
    def setup_advanced_data_generators(self):
        """Setup advanced data generators with confusion-focused augmentation"""
        print("📊 Setting up advanced data generators...")
        
        # Training augmentation - heavy for robustness
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            
            # Geometric augmentations
            rotation_range=25,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.15,
            zoom_range=0.2,
            horizontal_flip=True,
            
            # Color augmentations (important for fashion)
            brightness_range=[0.8, 1.2],
            channel_shift_range=0.1,
            
            # Advanced augmentations
            fill_mode='nearest'
        )
        
        # Validation/Test - minimal augmentation
        val_test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create generators with explicit class order
        train_generator = train_datagen.flow_from_directory(
            os.path.join(self.dataset_path, 'train'),
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical',
            shuffle=True,
            classes=self.categories,
            seed=42
        )
        
        validation_generator = val_test_datagen.flow_from_directory(
            os.path.join(self.dataset_path, 'validation'),
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical',
            shuffle=False,
            classes=self.categories,
            seed=42
        )
        
        test_generator = val_test_datagen.flow_from_directory(
            os.path.join(self.dataset_path, 'test'),
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical',
            shuffle=False,
            classes=self.categories,
            seed=42
        )
        
        print(f"✅ Data generators created:")
        print(f"   Train: {train_generator.samples} samples")
        print(f"   Validation: {validation_generator.samples} samples") 
        print(f"   Test: {test_generator.samples} samples")
        print(f"   Classes: {train_generator.class_indices}")
        
        return train_generator, validation_generator, test_generator
    
    def calculate_class_weights(self, train_generator):
        """Calculate class weights with emphasis on confused categories"""
        print("⚖️ Calculating class weights for balanced training...")
        
        # Get class distribution
        class_counts = {}
        for class_name, class_idx in train_generator.class_indices.items():
            class_counts[class_idx] = 0
        
        # Count samples per class
        for i in range(len(train_generator.filenames)):
            class_idx = train_generator.classes[i]
            class_counts[class_idx] += 1
        
        # Calculate base weights (inverse frequency)
        total_samples = sum(class_counts.values())
        base_weights = {}
        for class_idx, count in class_counts.items():
            base_weights[class_idx] = total_samples / (len(class_counts) * count)
        
        # Apply confusion-focused adjustments
        confusion_weights = {}
        for class_idx, class_name in enumerate(self.categories):
            weight = base_weights[class_idx]
            
            # Increase weight for categories that were confused in Step 1
            if class_name in ['dress', 'formal', 'outerwear']:  # 0% F1-score categories
                weight *= 3.0  # Triple the weight
            elif class_name in ['shoes', 'bottom']:  # Low F1-score categories
                weight *= 2.0  # Double the weight
            
            confusion_weights[class_idx] = weight
        
        print("Class weights:")
        for class_idx, weight in confusion_weights.items():
            class_name = self.categories[class_idx]
            print(f"   {class_name}: {weight:.2f}")
        
        return confusion_weights
    
    def setup_advanced_callbacks(self):
        """Setup comprehensive callbacks for training - FIXED serialization issues"""
        print("🔧 Setting up advanced callbacks...")
        
        callbacks = [
            # Model checkpointing - save best model (using .keras format)
            ModelCheckpoint(
                self.model_save_path,
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                mode='max',
                verbose=1
            ),
            
            # Early stopping with patience
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1,
                min_delta=0.001
            ),
            
            # Learning rate reduction
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.3,
                patience=7,
                min_lr=1e-8,
                verbose=1,
                cooldown=3
            ),
            
            # Fixed learning rate schedule - using function instead of lambda
            LearningRateScheduler(
                cosine_decay_schedule,
                verbose=0
            ),
            
            # Simplified TensorBoard logging
            TensorBoard(
                log_dir=f'logs/step3_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                histogram_freq=0,  # Reduced to avoid serialization issues
                write_graph=False,  # Disabled to avoid serialization issues
                write_images=False
            )
        ]
        
        return callbacks
    
    def compile_model_advanced(self, model):
        """Compile model with advanced optimization"""
        print("⚙️ Compiling model with advanced optimization...")
        
        # Advanced optimizer with learning rate scheduling
        optimizer = AdamW(
            learning_rate=0.001,
            weight_decay=0.0001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        
        # Compile with comprehensive metrics
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                TopKCategoricalAccuracy(k=2, name='top_2_accuracy'),
                TopKCategoricalAccuracy(k=3, name='top_3_accuracy'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        )
        
        print("✅ Model compiled with advanced metrics")
        return model
    
    def train_model_advanced(self, epochs=50):
        """Advanced training with multiple phases"""
        print("🚀 Starting advanced training process...")
        
        # Create enhanced model
        self.model = self.create_enhanced_model('efficientnet')
        
        # Setup data
        train_gen, val_gen, test_gen = self.setup_advanced_data_generators()
        
        # Calculate class weights
        class_weights = self.calculate_class_weights(train_gen)
        
        # Compile model
        self.model = self.compile_model_advanced(self.model)
        
        # Setup callbacks
        callbacks = self.setup_advanced_callbacks()
        
        print(f"\n🎯 Starting Phase 1: Feature Learning ({epochs//2} epochs)")
        print("Base model frozen, training classification head only...")
        
        # Phase 1: Train only classification head
        history_phase1 = self.model.fit(
            train_gen,
            epochs=epochs//2,
            validation_data=val_gen,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )
        
        print(f"\n🔥 Starting Phase 2: Fine-tuning ({epochs//2} epochs)")
        print("Unfreezing base model for fine-tuning...")
        
        # Phase 2: Unfreeze and fine-tune
        self.model.layers[4].trainable = True  # Unfreeze base model
        
        # Recompile with lower learning rate for fine-tuning
        optimizer_finetune = AdamW(
            learning_rate=0.0001,  # Lower learning rate
            weight_decay=0.0001
        )
        
        self.model.compile(
            optimizer=optimizer_finetune,
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                TopKCategoricalAccuracy(k=2, name='top_2_accuracy'),
                TopKCategoricalAccuracy(k=3, name='top_3_accuracy')
            ]
        )
        
        # Continue training with fine-tuning
        history_phase2 = self.model.fit(
            train_gen,
            epochs=epochs,
            initial_epoch=epochs//2,
            validation_data=val_gen,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )
        
        # Combine histories - FIXED to handle missing keys
        self.training_history = {
            'phase1': history_phase1.history,
            'phase2': history_phase2.history,
            'combined': self._combine_histories(history_phase1.history, history_phase2.history)
        }
        
        print("✅ Advanced training completed!")
        
        # Evaluate on test set
        print("\n📊 Evaluating on test set...")
        test_results = self.model.evaluate(test_gen, verbose=1)
        
        print(f"🎯 Final Test Results:")
        for i, metric_name in enumerate(self.model.metrics_names):
            print(f"   {metric_name}: {test_results[i]:.4f}")
        
        return self.training_history, test_results
    
    def _combine_histories(self, hist1, hist2):
        """Combine training histories from two phases - FIXED to handle missing keys"""
        combined = {}
        all_keys = set(hist1.keys()).union(set(hist2.keys()))
        for key in all_keys:
            vals1 = hist1.get(key, [])
            vals2 = hist2.get(key, [])
            combined[key] = vals1 + vals2
        return combined
    
    def evaluate_model_comprehensive(self, test_generator):
        """Comprehensive model evaluation and comparison with baseline"""
        print("🔍 Performing comprehensive model evaluation...")
        
        # Get predictions
        predictions = self.model.predict(test_generator, verbose=1)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = test_generator.classes
        
        # Classification report
        class_report = classification_report(
            true_classes, predicted_classes,
            target_names=self.categories,
            output_dict=True
        )
        
        print("\n📊 Detailed Classification Report:")
        print(classification_report(true_classes, predicted_classes, target_names=self.categories))
        
        # Confusion matrix
        cm = confusion_matrix(true_classes, predicted_classes)
        
        # Compare with Step 1 baseline
        self._compare_with_baseline(class_report, cm)
        
        # Save comprehensive results
        results = {
            'timestamp': datetime.now().isoformat(),
            'model_path': self.model_save_path,
            'dataset_path': self.dataset_path,
            'total_test_samples': len(true_classes),
            'overall_accuracy': float(np.mean(predicted_classes == true_classes)),
            'classification_report': class_report,
            'confusion_matrix': cm.tolist(),
            'training_history': self.training_history,
            'improvements_vs_baseline': self._calculate_improvements(class_report)
        }
        
        # Save results
        with open('step3_training_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        return results
    
    def _compare_with_baseline(self, new_report, new_cm):
        """Compare with Step 1 baseline results"""
        print("\n📈 IMPROVEMENT ANALYSIS vs Step 1 Baseline:")
        print("=" * 60)
        
        # Baseline results from Step 1
        baseline_accuracy = 0.333
        baseline_f1_scores = {
            'bottom': 0.4, 'dress': 0.0, 'formal': 0.0,
            'outerwear': 0.0, 'shoes': 0.461, 'top': 0.571
        }
        
        # Current results
        current_accuracy = new_report['accuracy']
        
        print(f"Overall Accuracy:")
        print(f"   Baseline: {baseline_accuracy:.1%}")
        print(f"   Current:  {current_accuracy:.1%}")
        print(f"   Improvement: {((current_accuracy - baseline_accuracy) / baseline_accuracy * 100):+.1f}%")
        print()
        
        print("Per-Category F1-Score Improvements:")
        for category in self.categories:
            if category in new_report:
                baseline_f1 = baseline_f1_scores[category]
                current_f1 = new_report[category]['f1-score']
                
                if baseline_f1 == 0:
                    improvement = "∞" if current_f1 > 0 else "No change"
                else:
                    improvement = f"{((current_f1 - baseline_f1) / baseline_f1 * 100):+.1f}%"
                
                print(f"   {category:12}: {baseline_f1:.3f} → {current_f1:.3f} ({improvement})")
    
    def _calculate_improvements(self, class_report):
        """Calculate specific improvements achieved"""
        improvements = {
            'eliminated_zero_f1_categories': [],
            'significant_improvements': [],
            'overall_metrics': {
                'accuracy_improvement': 'TBD',
                'avg_f1_improvement': 'TBD',
                'worst_category_improvement': 'TBD'
            }
        }
        
        # Check for eliminated zero F1-scores
        baseline_zeros = ['dress', 'formal', 'outerwear']
        for category in baseline_zeros:
            if category in class_report and class_report[category]['f1-score'] > 0:
                improvements['eliminated_zero_f1_categories'].append(category)
        
        return improvements
    
    def generate_visualizations(self, results):
        """Generate comprehensive training and evaluation visualizations"""
        print("📈 Generating comprehensive visualizations...")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Training History
        plt.subplot(3, 3, 1)
        history = results['training_history']['combined']
        plt.plot(history['accuracy'], label='Training Accuracy', linewidth=2)
        plt.plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        plt.title('Training Progress', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 3, 2)
        plt.plot(history['loss'], label='Training Loss', linewidth=2)
        plt.plot(history['val_loss'], label='Validation Loss', linewidth=2)
        plt.title('Loss Progress', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Confusion Matrix
        plt.subplot(3, 3, 3)
        cm = np.array(results['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.categories, yticklabels=self.categories)
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        # 3. Per-Class Performance
        plt.subplot(3, 3, 4)
        class_report = results['classification_report']
        categories = [cat for cat in self.categories if cat in class_report]
        f1_scores = [class_report[cat]['f1-score'] for cat in categories]
        
        bars = plt.bar(categories, f1_scores, 
                      color=['red' if f1 < 0.5 else 'orange' if f1 < 0.7 else 'green' for f1 in f1_scores])
        plt.title('F1-Score by Category', fontsize=14, fontweight='bold')
        plt.ylabel('F1-Score')
        plt.xticks(rotation=45)
        plt.axhline(y=0.7, color='orange', linestyle='--', alpha=0.7)
        plt.axhline(y=0.8, color='green', linestyle='--', alpha=0.7)
        
        # Add value labels on bars
        for bar, f1 in zip(bars, f1_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{f1:.3f}', ha='center', va='bottom')
        
        # 4. Precision vs Recall
        plt.subplot(3, 3, 5)
        precisions = [class_report[cat]['precision'] for cat in categories]
        recalls = [class_report[cat]['recall'] for cat in categories]
        
        plt.scatter(precisions, recalls, s=100, c=f1_scores, cmap='RdYlGn', alpha=0.7)
        for i, cat in enumerate(categories):
            plt.annotate(cat, (precisions[i], recalls[i]), xytext=(5, 5), 
                        textcoords='offset points', fontsize=8)
        plt.xlabel('Precision')
        plt.ylabel('Recall')
        plt.title('Precision vs Recall', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # 5. Top-K Accuracy
        if 'top_2_accuracy' in history:
            plt.subplot(3, 3, 6)
            plt.plot(history['accuracy'], label='Top-1 (Accuracy)', linewidth=2)
            plt.plot(history['top_2_accuracy'], label='Top-2', linewidth=2)
            if 'top_3_accuracy' in history:
                plt.plot(history['top_3_accuracy'], label='Top-3', linewidth=2)
            plt.title('Top-K Accuracy', fontsize=14, fontweight='bold')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 6. Learning Rate (if available)
        plt.subplot(3, 3, 7)
        if 'lr' in history:
            plt.plot(history['lr'], linewidth=2)
            plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.yscale('log')
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, 'Learning Rate\nData Not Available', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        
        # 7. Model Metrics Summary
        plt.subplot(3, 3, 8)
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        values = [
            results['overall_accuracy'],
            class_report['macro avg']['precision'],
            class_report['macro avg']['recall'],
            class_report['macro avg']['f1-score']
        ]
        
        bars = plt.bar(metrics, values, color=['lightblue', 'lightgreen', 'lightcoral', 'lightyellow'])
        plt.title('Overall Performance Metrics', fontsize=14, fontweight='bold')
        plt.ylabel('Score')
        plt.ylim(0, 1)
        
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # 8. Improvement vs Baseline
        plt.subplot(3, 3, 9)
        baseline_f1s = [0.4, 0.0, 0.0, 0.0, 0.461, 0.571]  # From Step 1
        current_f1s = [class_report[cat]['f1-score'] if cat in class_report else 0 for cat in self.categories]
        
        x = range(len(self.categories))
        width = 0.35
        
        plt.bar([i - width/2 for i in x], baseline_f1s, width, label='Baseline (Step 1)', alpha=0.7)
        plt.bar([i + width/2 for i in x], current_f1s, width, label='Current Model', alpha=0.7)
        
        plt.title('Improvement vs Baseline', fontsize=14, fontweight='bold')
        plt.xlabel('Categories')
        plt.ylabel('F1-Score')
        plt.xticks(x, self.categories, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('step3_comprehensive_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Visualizations saved as 'step3_comprehensive_results.png'")

# Main execution function
def run_step3_advanced_training():
    """Execute Step 3: Advanced Model Retraining"""
    
    print("🚀 STEP 3: ADVANCED MODEL RETRAINING")
    print("=" * 60)
    print("Training enhanced model with improved dataset from Step 2...")
    print()
    
    # Initialize trainer
    trainer = AdvancedModelTrainer()
    
    # Train model
    training_history, test_results = trainer.train_model_advanced(epochs=40)
    
    # Setup test data for evaluation
    _, _, test_gen = trainer.setup_advanced_data_generators()
    
    # Comprehensive evaluation
    results = trainer.evaluate_model_comprehensive(test_gen)
    
    # Generate visualizations
    trainer.generate_visualizations(results)
    
    print("\n✅ STEP 3 ADVANCED TRAINING COMPLETE!")
    print("=" * 60)
    print(f"🤖 Model saved: {trainer.model_save_path}")
    print(f"📊 Results saved: step3_training_results.json")
    print(f"📈 Visualizations: step3_comprehensive_results.png")
    print(f"🎯 Expected improvement: 33% → 75%+ accuracy")
    
    return trainer.model, results

# Execute Step 3
if __name__ == "__main__":
    model, results = run_step3_advanced_training()
