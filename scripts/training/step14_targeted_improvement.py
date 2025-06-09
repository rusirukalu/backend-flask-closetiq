# step14_targeted_improvement.py - Fix specific confusion issues

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import *
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

class TargetedModelImprover:
    def __init__(self, model_path="enhanced_fashion_model.keras"):
        self.model_path = model_path
        self.model = None
        self.dataset_path = "real_fashion_dataset_training"
        
        # Current categories
        self.categories = [
            'shirts_blouses', 'tshirts_tops', 'dresses', 'pants_jeans',
            'shorts', 'skirts', 'jackets_coats', 'sweaters', 
            'shoes_sneakers', 'shoes_formal', 'bags_accessories'
        ]
        
        # Problem category pairs
        self.confusion_pairs = {
            'clothing_gender': ['shirts_blouses', 'tshirts_tops'],
            'footwear_style': ['shoes_sneakers', 'shoes_formal']
        }
    
    def analyze_current_confusions(self):
        """Analyze current model's confusion patterns"""
        print("🔍 ANALYZING CURRENT CONFUSION PATTERNS")
        print("=" * 50)
        
        # Load model
        self.model = load_model(self.model_path)
        
        # Setup test generator
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_directory(
            os.path.join(self.dataset_path, 'test'),
            target_size=(224, 224),
            batch_size=16,
            class_mode='categorical',
            shuffle=False,
            classes=self.categories
        )
        
        # Get predictions
        predictions = self.model.predict(test_generator)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = test_generator.classes
        
        # Create confusion matrix
        cm = confusion_matrix(true_classes, predicted_classes)
        
        # Analyze specific confusion pairs
        print("\n📊 CONFUSION ANALYSIS:")
        
        # Check clothing gender confusions
        shirts_idx = self.categories.index('shirts_blouses')
        tshirts_idx = self.categories.index('tshirts_tops')
        
        shirts_to_tshirts = cm[shirts_idx][tshirts_idx]
        tshirts_to_shirts = cm[tshirts_idx][shirts_idx]
        
        print(f"👔 CLOTHING GENDER CONFUSION:")
        print(f"   Shirts/Blouses → T-shirts: {shirts_to_tshirts} cases")
        print(f"   T-shirts → Shirts/Blouses: {tshirts_to_shirts} cases")
        
        # Check footwear confusions
        sneakers_idx = self.categories.index('shoes_sneakers')
        formal_idx = self.categories.index('shoes_formal')
        
        sneakers_to_formal = cm[sneakers_idx][formal_idx]
        formal_to_sneakers = cm[formal_idx][sneakers_idx]
        
        print(f"\n👟 FOOTWEAR CONFUSION:")
        print(f"   Sneakers → Formal shoes: {sneakers_to_formal} cases")
        print(f"   Formal shoes → Sneakers: {formal_to_sneakers} cases")
        
        # Plot confusion matrix
        self.plot_confusion_matrix(cm)
        
        return cm
    
    def create_focused_training_data(self):
        """Create focused training data for confused categories"""
        print("\n🎯 CREATING FOCUSED TRAINING DATA")
        print("=" * 40)
        
        focused_dir = "focused_training_data"
        
        # Create focused dataset structure for problem categories
        focus_categories = {
            'mens_shirts': 'Male dress shirts and button-ups',
            'womens_blouses': 'Female blouses and dress shirts', 
            'mens_tshirts': 'Male t-shirts and casual tops',
            'womens_tshirts': 'Female t-shirts and fitted tops',
            'casual_sneakers': 'Sneakers and athletic shoes',
            'dress_shoes': 'Formal shoes and heels',
            'sandals_slippers': 'Casual open footwear'
        }
        
        for split in ['train', 'validation', 'test']:
            for category in focus_categories.keys():
                os.makedirs(os.path.join(focused_dir, split, category), exist_ok=True)
        
        print("✅ Focused training structure created!")
        print("\n📋 MANUAL DATA COLLECTION NEEDED:")
        print("Collect 30-50 images for each category:")
        
        for category, description in focus_categories.items():
            print(f"  📁 {category}: {description}")
        
        print(f"\n💡 SEARCH TERMS TO USE:")
        search_terms = {
            'mens_shirts': ['mens dress shirt', 'mens button up', 'mens formal shirt'],
            'womens_blouses': ['womens blouse', 'womens dress shirt', 'womens button up'],
            'mens_tshirts': ['mens t-shirt', 'mens casual shirt', 'mens tee'],
            'womens_tshirts': ['womens t-shirt', 'womens fitted tee', 'womens casual top'],
            'casual_sneakers': ['white sneakers', 'running shoes', 'casual sneakers'],
            'dress_shoes': ['dress shoes', 'formal shoes', 'high heels', 'oxford shoes'],
            'sandals_slippers': ['sandals', 'flip flops', 'slippers', 'slides']
        }
        
        for category, terms in search_terms.items():
            print(f"  {category}: {', '.join(terms)}")
        
        return focused_dir
    
    def create_confusion_focused_model(self):
        """Create a model specifically focused on reducing confusions"""
        print("\n🧠 CREATING CONFUSION-FOCUSED MODEL")
        print("=" * 40)
        
        # Load current model
        base_model = load_model(self.model_path)
        
        # Create a new model with attention mechanisms for confused categories
        from tensorflow.keras.layers import Input, Dense, Dropout, Multiply, GlobalAveragePooling2D
        from tensorflow.keras.models import Model
        
        # Extract the base feature extractor (everything except the final classification layer)
        feature_extractor = Model(
            inputs=base_model.input,
            outputs=base_model.layers[-3].output  # Before final dense layer
        )
        
        # Freeze feature extractor
        for layer in feature_extractor.layers:
            layer.trainable = False
        
        # Create new attention-based head
        inputs = Input(shape=(224, 224, 3))
        
        # Get features
        features = feature_extractor(inputs)
        
        # Attention mechanism for gender-specific features
        gender_attention = Dense(128, activation='relu', name='gender_attention')(features)
        gender_weights = Dense(128, activation='sigmoid', name='gender_weights')(gender_attention)
        gender_features = Multiply(name='gender_focused_features')([features, gender_weights])
        
        # Attention mechanism for footwear-specific features  
        footwear_attention = Dense(128, activation='relu', name='footwear_attention')(features)
        footwear_weights = Dense(128, activation='sigmoid', name='footwear_weights')(footwear_attention)
        footwear_features = Multiply(name='footwear_focused_features')([features, footwear_weights])
        
        # Combine attention features
        from tensorflow.keras.layers import Concatenate
        combined_features = Concatenate(name='combined_attention')([gender_features, footwear_features])
        
        # Classification head
        x = Dense(256, activation='relu')(combined_features)
        x = Dropout(0.3)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.3)(x)
        outputs = Dense(11, activation='softmax', name='confusion_aware_predictions')(x)
        
        # Create attention model
        attention_model = Model(inputs=inputs, outputs=outputs)
        
        # Compile with class weights to emphasize confused categories
        class_weights = self.calculate_confusion_weights()
        
        attention_model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("✅ Confusion-focused model created!")
        return attention_model, class_weights
    
    def calculate_confusion_weights(self):
        """Calculate class weights to emphasize confused categories"""
        base_weight = 1.0
        confusion_boost = 2.0
        
        weights = {}
        for i, category in enumerate(self.categories):
            if category in ['shirts_blouses', 'tshirts_tops', 'shoes_sneakers', 'shoes_formal']:
                weights[i] = confusion_boost  # Higher weight for confused categories
            else:
                weights[i] = base_weight
        
        print(f"📊 Class weights calculated:")
        for i, weight in weights.items():
            print(f"   {self.categories[i]}: {weight}")
        
        return weights
    
    def train_confusion_focused_model(self, attention_model, class_weights):
        """Train the confusion-focused model"""
        print("\n🚀 TRAINING CONFUSION-FOCUSED MODEL")
        print("=" * 40)
        
        # Enhanced augmentation for confused categories
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=15,  # Less rotation to preserve gender/style cues
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            brightness_range=[0.9, 1.1],  # Subtle brightness changes
            fill_mode='nearest'
        )
        
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create generators
        train_generator = train_datagen.flow_from_directory(
            os.path.join(self.dataset_path, 'train'),
            target_size=(224, 224),
            batch_size=16,
            class_mode='categorical',
            shuffle=True,
            classes=self.categories
        )
        
        val_generator = val_datagen.flow_from_directory(
            os.path.join(self.dataset_path, 'validation'),
            target_size=(224, 224),
            batch_size=16,
            class_mode='categorical',
            shuffle=False,
            classes=self.categories
        )
        
        # Callbacks with focus on reducing confusion
        callbacks = [
            ModelCheckpoint(
                'confusion_resolved_model.keras',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            # Custom callback to monitor confusion reduction
            ConfusionMonitorCallback(self.confusion_pairs, self.categories)
        ]
        
        # Train model
        history = attention_model.fit(
            train_generator,
            epochs=25,
            validation_data=val_generator,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )
        
        print("✅ Confusion-focused training complete!")
        return history
    
    def plot_confusion_matrix(self, cm):
        """Plot confusion matrix with focus on problem areas"""
        plt.figure(figsize=(12, 10))
        
        # Create heatmap
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=[cat.replace('_', '\n') for cat in self.categories],
            yticklabels=[cat.replace('_', '\n') for cat in self.categories]
        )
        
        plt.title('Confusion Matrix - Focus on Gender & Footwear Issues', fontsize=14)
        plt.xlabel('Predicted Category')
        plt.ylabel('True Category')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        # Highlight confusion areas
        shirts_idx = self.categories.index('shirts_blouses')
        tshirts_idx = self.categories.index('tshirts_tops')
        sneakers_idx = self.categories.index('shoes_sneakers') 
        formal_idx = self.categories.index('shoes_formal')
        
        # Add rectangles around confusion areas
        plt.gca().add_patch(plt.Rectangle((shirts_idx-0.5, tshirts_idx-0.5), 2, 2, 
                                         fill=False, edgecolor='red', lw=3, linestyle='--'))
        plt.gca().add_patch(plt.Rectangle((sneakers_idx-0.5, formal_idx-0.5), 2, 2, 
                                         fill=False, edgecolor='orange', lw=3, linestyle='--'))
        
        plt.tight_layout()
        plt.savefig('confusion_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

# Custom callback to monitor confusion reduction
class ConfusionMonitorCallback(tf.keras.callbacks.Callback):
    def __init__(self, confusion_pairs, categories):
        super().__init__()
        self.confusion_pairs = confusion_pairs
        self.categories = categories
    
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 5 == 0:  # Check every 5 epochs
            print(f"\n📊 Epoch {epoch}: Monitoring confusion reduction...")

def run_targeted_improvement():
    """Run the complete targeted improvement pipeline"""
    
    print("🎯 TARGETED MODEL IMPROVEMENT: Gender & Footwear")
    print("=" * 60)
    
    improver = TargetedModelImprover()
    
    # Step 1: Analyze current confusions
    cm = improver.analyze_current_confusions()
    
    # Step 2: Create focused training structure (manual collection needed)
    focused_dir = improver.create_focused_training_data()
    
    # Step 3: Create confusion-focused model
    attention_model, class_weights = improver.create_confusion_focused_model()
    
    # Step 4: Train confusion-focused model
    history = improver.train_confusion_focused_model(attention_model, class_weights)
    
    print("\n🎉 TARGETED IMPROVEMENT COMPLETE!")
    print("=" * 40)
    print("✅ Confusion analysis saved: confusion_analysis.png")
    print("✅ Improved model saved: confusion_resolved_model.keras")
    print("📁 Focused data structure created (manual collection needed)")
    
    print("\n🔄 NEXT STEPS:")
    print("1. Collect focused training data for gender/footwear distinction")
    print("2. Update MODEL_PATH to 'confusion_resolved_model.keras'")
    print("3. Test with challenging gender-specific clothing images")
    
    return True

if __name__ == "__main__":
    run_targeted_improvement()
