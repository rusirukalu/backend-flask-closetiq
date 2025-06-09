# step3_multilabel_attribute_prediction_fixed.py - Fixed Multi-label Attribute Prediction

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Concatenate, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, hamming_loss, jaccard_score
import json
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import cv2
from PIL import Image

class MultiLabelAttributePredictor:
    """Advanced Multi-label Fashion Attribute Prediction System"""
    
    def __init__(self, base_model_path='final_enhanced_model.keras'):
        self.base_model_path = base_model_path
        self.base_model = None
        self.feature_extractor = None
        self.attribute_models = {}
        self.label_encoders = {}
        
        # Enhanced attribute configuration with realistic combinations
        self.attributes_config = {
            'colors': {
                'classes': ['black', 'white', 'gray', 'brown', 'red', 'pink', 'orange', 
                           'yellow', 'green', 'blue', 'purple', 'beige', 'navy', 'multicolor'],
                'type': 'multi_label',
                'max_labels': 3,  # Maximum 3 colors per item
                'description': 'Primary and secondary colors'
            },
            'patterns': {
                'classes': ['solid', 'striped', 'checkered', 'floral', 'geometric', 
                           'abstract', 'animal_print', 'polka_dots', 'plaid', 'paisley'],
                'type': 'multi_label',
                'max_labels': 2,  # Can have mixed patterns
                'description': 'Visual patterns and designs'
            },
            'style_attributes': {
                'classes': ['casual', 'formal', 'vintage', 'modern', 'bohemian', 
                           'minimalist', 'edgy', 'romantic', 'sporty', 'classic'],
                'type': 'multi_label',
                'max_labels': 3,  # Multiple style influences
                'description': 'Overall style characteristics'
            },
            'material_properties': {
                'classes': ['lightweight', 'heavy', 'stretchy', 'rigid', 'breathable', 
                           'waterproof', 'warm', 'cool', 'structured', 'flowy'],
                'type': 'multi_label',
                'max_labels': 4,  # Multiple material properties
                'description': 'Material characteristics and properties'
            },
            'occasion_suitability': {
                'classes': ['work', 'casual_day', 'date_night', 'party', 'formal_event', 
                           'workout', 'beach', 'travel', 'home', 'outdoor'],
                'type': 'multi_label',
                'max_labels': 5,  # Very versatile items
                'description': 'Suitable occasions for wearing'
            },
            'seasonal_appropriate': {
                'classes': ['spring', 'summer', 'fall', 'winter', 'transitional'],
                'type': 'multi_label',
                'max_labels': 3,  # Good for multiple seasons
                'description': 'Seasonal appropriateness'
            },
            'care_instructions': {
                'classes': ['machine_wash', 'hand_wash', 'dry_clean', 'air_dry', 
                           'tumble_dry', 'iron', 'steam', 'delicate'],
                'type': 'multi_label',
                'max_labels': 4,  # Multiple care requirements
                'description': 'Garment care requirements'
            },
            'size_fit': {
                'classes': ['runs_small', 'true_to_size', 'runs_large', 'stretchy_fit', 
                           'fitted', 'loose', 'oversized', 'adjustable'],
                'type': 'multi_label',
                'max_labels': 2,  # Size and fit characteristics
                'description': 'Size and fit characteristics'
            }
        }
        
        # Base categories
        self.base_categories = [
            'shirts_blouses', 'tshirts_tops', 'dresses', 'pants_jeans',
            'shorts', 'skirts', 'jackets_coats', 'sweaters', 
            'shoes_sneakers', 'shoes_formal', 'bags_accessories'
        ]
        
        # Training configuration
        self.training_config = {
            'epochs': 30,
            'batch_size': 16,
            'learning_rate': 0.001,
            'validation_split': 0.2,
            'early_stopping_patience': 8,
            'reduce_lr_patience': 4
        }
    
    def load_base_model_and_create_feature_extractor(self):
        """Load base model and create feature extractor"""
        print("🔧 LOADING BASE MODEL FOR ATTRIBUTE PREDICTION")
        print("=" * 60)
        
        try:
            # Load excellent base model
            self.base_model = load_model(self.base_model_path)
            print(f"✅ Base model loaded: {self.base_model_path}")
            
            # Create feature extractor (before final classification)
            feature_layer = self.base_model.layers[-3]
            self.feature_extractor = Model(
                inputs=self.base_model.input,
                outputs=feature_layer.output,
                name='multilabel_feature_extractor'
            )
            
            print(f"✅ Feature extractor created:")
            print(f"   Feature dimensions: {self.feature_extractor.output_shape[-1]}")
            print(f"   Will be used for all attribute predictions")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading base model: {e}")
            return False
    
    def create_advanced_multilabel_model(self, attribute_name):
        """Create advanced multi-label classification model"""
        print(f"\n🏗️ Creating advanced model for: {attribute_name}")
        
        config = self.attributes_config[attribute_name]
        num_classes = len(config['classes'])
        max_labels = config['max_labels']
        
        # Input layer
        inputs = Input(shape=(224, 224, 3))
        
        # Extract features using pre-trained base
        base_features = self.feature_extractor(inputs)
        
        # Ensure we have the right shape
        if len(base_features.shape) > 2:
            features = GlobalAveragePooling2D()(base_features)
        else:
            features = base_features
        
        # Advanced multi-label architecture
        # Branch 1: Global context
        global_branch = Dense(512, activation='relu', name=f'{attribute_name}_global_1')(features)
        global_branch = BatchNormalization()(global_branch)
        global_branch = Dropout(0.3)(global_branch)
        global_branch = Dense(256, activation='relu', name=f'{attribute_name}_global_2')(global_branch)
        global_branch = Dropout(0.4)(global_branch)
        
        # Branch 2: Attribute-specific features
        attr_branch = Dense(256, activation='relu', name=f'{attribute_name}_attr_1')(features)
        attr_branch = BatchNormalization()(attr_branch)
        attr_branch = Dropout(0.3)(attr_branch)
        attr_branch = Dense(128, activation='relu', name=f'{attribute_name}_attr_2')(attr_branch)
        attr_branch = Dropout(0.3)(attr_branch)
        
        # Combine branches
        combined = Concatenate(name=f'{attribute_name}_combined')([global_branch, attr_branch])
        combined = Dense(256, activation='relu')(combined)
        combined = BatchNormalization()(combined)
        combined = Dropout(0.4)(combined)
        
        # Multi-label output layer with sigmoid activation
        outputs = Dense(num_classes, activation='sigmoid', 
                       name=f'{attribute_name}_multilabel_output')(combined)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs, 
                     name=f'{attribute_name}_multilabel_classifier')
        
        # Custom loss for multi-label with class balancing
        def balanced_binary_crossentropy(y_true, y_pred):
            # Add small epsilon to prevent log(0)
            epsilon = tf.keras.backend.epsilon()
            y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
            
            # Calculate positive and negative weights
            pos_weight = tf.reduce_sum(1 - y_true) / tf.reduce_sum(y_true + epsilon)
            
            # Weighted binary crossentropy
            loss = -(pos_weight * y_true * tf.math.log(y_pred) + 
                    (1 - y_true) * tf.math.log(1 - y_pred))
            
            return tf.reduce_mean(loss)
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.training_config['learning_rate']),
            loss=balanced_binary_crossentropy,
            metrics=['binary_accuracy', 'precision', 'recall']
        )
        
        print(f"   ✅ {attribute_name} multi-label model created")
        print(f"      Classes: {num_classes}")
        print(f"      Max labels: {max_labels}")
        print(f"      Parameters: {model.count_params():,}")
        
        return model
    
    def safe_random_choice(self, options, min_size, max_size, replace=False):
        """Safely choose random items without exceeding population size"""
        available_size = len(options)
        
        if replace:
            # Can sample with replacement
            actual_size = np.random.randint(min_size, max_size + 1)
        else:
            # Can't sample more than available without replacement
            max_possible = min(max_size, available_size)
            min_possible = min(min_size, available_size)
            actual_size = np.random.randint(min_possible, max_possible + 1)
        
        if actual_size == 0:
            return []
        
        return list(np.random.choice(options, size=actual_size, replace=replace))
    
    def generate_realistic_multilabel_data(self, num_samples=2000):
        """Generate realistic multi-label training data"""
        print(f"\n🎨 GENERATING REALISTIC MULTI-LABEL TRAINING DATA")
        print("=" * 60)
        
        training_samples = []
        
        # Category-based attribute generation with realistic combinations
        for i in range(num_samples):
            # Random base category
            base_category = np.random.choice(self.base_categories)
            
            sample = {
                'sample_id': f'ml_sample_{i}',
                'base_category': base_category,
                'multilabel_attributes': {}
            }
            
            # Generate realistic multi-label attributes based on category
            if 'shirts' in base_category or 'tshirts' in base_category:
                sample['multilabel_attributes'] = self._generate_top_multilabels()
            elif 'dresses' in base_category:
                sample['multilabel_attributes'] = self._generate_dress_multilabels()
            elif 'pants' in base_category or 'jeans' in base_category:
                sample['multilabel_attributes'] = self._generate_pants_multilabels()
            elif 'shoes' in base_category:
                sample['multilabel_attributes'] = self._generate_shoes_multilabels()
            elif 'jackets' in base_category:
                sample['multilabel_attributes'] = self._generate_jacket_multilabels()
            else:
                sample['multilabel_attributes'] = self._generate_general_multilabels()
            
            training_samples.append(sample)
        
        print(f"✅ Generated {len(training_samples)} realistic multi-label samples")
        
        # Save training data
        with open('multilabel_training_data.json', 'w') as f:
            json.dump(training_samples, f, indent=2)
        
        # Analyze label distributions
        self._analyze_label_distributions(training_samples)
        
        return training_samples
    
    def _generate_top_multilabels(self):
        """Generate multi-labels for tops with realistic combinations - FIXED"""
        # Colors: 1-3 colors common for tops
        colors = self.safe_random_choice(['black', 'white', 'blue', 'gray', 'red', 'navy', 'green'], 1, 3)
        
        # Patterns: Usually 1, sometimes 2 (e.g., striped with floral details)
        patterns = self.safe_random_choice(['solid', 'striped', 'checkered'], 1, 2)
        
        # Style: Multiple style influences common
        style_pool = ['casual', 'modern', 'classic', 'minimalist']
        if np.random.random() > 0.3:  # 70% chance of formal influence
            style_pool.extend(['formal', 'professional'])
        style = self.safe_random_choice(style_pool, 1, 3)
        
        # Material properties: Multiple properties realistic
        material = self.safe_random_choice(['lightweight', 'breathable', 'stretchy', 'structured'], 2, 4)
        
        # Occasions: Tops are versatile - FIXED
        occasion_base = ['casual_day', 'work']
        if 'formal' in style:
            occasion_base.extend(['formal_event', 'date_night'])
        occasions = self.safe_random_choice(occasion_base, 1, min(len(occasion_base), 4))
        
        # Seasons: Most tops work multiple seasons
        seasons = self.safe_random_choice(['spring', 'summer', 'fall', 'transitional'], 2, 3)
        
        # Care: Realistic care combinations
        care = ['machine_wash', 'tumble_dry']
        if np.random.random() > 0.7:
            care.append('iron')
        
        # Size/fit
        size_fit = self.safe_random_choice(['true_to_size', 'fitted', 'stretchy_fit'], 1, 2)
        
        return {
            'colors': colors,
            'patterns': patterns,
            'style_attributes': style,
            'material_properties': material,
            'occasion_suitability': occasions,
            'seasonal_appropriate': seasons,
            'care_instructions': care,
            'size_fit': size_fit
        }
    
    def _generate_dress_multilabels(self):
        """Generate multi-labels for dresses - FIXED"""
        colors = self.safe_random_choice(['black', 'red', 'blue', 'white', 'pink', 'navy'], 1, 3)
        
        patterns = self.safe_random_choice(['solid', 'floral', 'geometric', 'abstract'], 1, 2)
        
        # Dresses often have multiple style influences
        style = self.safe_random_choice(['romantic', 'formal', 'casual', 'vintage', 'modern'], 2, 3)
        
        material = self.safe_random_choice(['flowy', 'lightweight', 'structured', 'stretchy'], 2, 3)
        
        # Dresses are versatile for occasions
        occasions = self.safe_random_choice(['date_night', 'party', 'formal_event', 'casual_day', 'work'], 2, 4)
        
        seasons = self.safe_random_choice(['spring', 'summer', 'transitional'], 1, 3)
        
        care = ['hand_wash', 'air_dry']
        if np.random.random() > 0.5:
            care.append('steam')
        
        size_fit = self.safe_random_choice(['fitted', 'true_to_size', 'flowy'], 1, 2)
        
        return {
            'colors': colors,
            'patterns': patterns,
            'style_attributes': style,
            'material_properties': material,
            'occasion_suitability': occasions,
            'seasonal_appropriate': seasons,
            'care_instructions': care,
            'size_fit': size_fit
        }
    
    def _generate_pants_multilabels(self):
        """Generate multi-labels for pants/jeans - FIXED"""
        colors = self.safe_random_choice(['black', 'blue', 'gray', 'brown', 'navy'], 1, 2)
        
        patterns = ['solid']  # Pants usually solid
        
        style = self.safe_random_choice(['casual', 'classic', 'modern', 'formal'], 1, 3)
        
        material = self.safe_random_choice(['stretchy', 'structured', 'heavy', 'rigid'], 2, 3)
        
        occasions = self.safe_random_choice(['casual_day', 'work', 'travel', 'outdoor'], 2, 4)
        
        seasons = ['fall', 'winter', 'spring', 'transitional']  # Pants work most seasons
        
        care = ['machine_wash', 'tumble_dry']
        
        size_fit = self.safe_random_choice(['true_to_size', 'fitted', 'stretchy_fit'], 1, 2)
        
        return {
            'colors': colors,
            'patterns': patterns,
            'style_attributes': style,
            'material_properties': material,
            'occasion_suitability': occasions,
            'seasonal_appropriate': seasons,
            'care_instructions': care,
            'size_fit': size_fit
        }
    
    def _generate_shoes_multilabels(self):
        """Generate multi-labels for shoes - FIXED"""
        colors = self.safe_random_choice(['black', 'brown', 'white', 'navy'], 1, 2)
        
        patterns = ['solid']  # Shoes usually solid
        
        style = self.safe_random_choice(['casual', 'formal', 'sporty', 'classic'], 1, 2)
        
        material = self.safe_random_choice(['waterproof', 'breathable', 'structured', 'rigid'], 2, 3)
        
        occasions = self.safe_random_choice(['work', 'casual_day', 'formal_event', 'outdoor', 'workout'], 1, 3)
        
        seasons = self.safe_random_choice(['spring', 'summer', 'fall', 'winter'], 2, 4)
        
        care = ['air_dry', 'delicate']
        
        size_fit = self.safe_random_choice(['true_to_size', 'runs_small'], 1, 1)
        
        return {
            'colors': colors,
            'patterns': patterns,
            'style_attributes': style,
            'material_properties': material,
            'occasion_suitability': occasions,
            'seasonal_appropriate': seasons,
            'care_instructions': care,
            'size_fit': size_fit
        }
    
    def _generate_jacket_multilabels(self):
        """Generate multi-labels for jackets/coats - FIXED"""
        colors = self.safe_random_choice(['black', 'navy', 'gray', 'brown', 'beige'], 1, 2)
        
        patterns = self.safe_random_choice(['solid', 'checkered'], 1, 1)
        
        style = self.safe_random_choice(['classic', 'formal', 'casual', 'edgy'], 1, 3)
        
        material = self.safe_random_choice(['warm', 'waterproof', 'structured', 'heavy'], 2, 4)
        
        occasions = self.safe_random_choice(['work', 'outdoor', 'formal_event', 'travel'], 2, 4)
        
        seasons = self.safe_random_choice(['fall', 'winter', 'spring'], 1, 3)
        
        care = ['dry_clean', 'air_dry']
        
        size_fit = self.safe_random_choice(['true_to_size', 'oversized'], 1, 1)
        
        return {
            'colors': colors,
            'patterns': patterns,
            'style_attributes': style,
            'material_properties': material,
            'occasion_suitability': occasions,
            'seasonal_appropriate': seasons,
            'care_instructions': care,
            'size_fit': size_fit
        }
    
    def _generate_general_multilabels(self):
        """Generate general multi-labels for other items - FIXED"""
        colors = self.safe_random_choice(['black', 'brown', 'gray'], 1, 2)
        
        patterns = ['solid']
        
        style = self.safe_random_choice(['casual', 'classic'], 1, 1)
        
        material = self.safe_random_choice(['lightweight', 'structured'], 1, 2)
        
        occasions = ['casual_day']
        
        seasons = ['spring', 'summer', 'fall', 'winter']
        
        care = ['machine_wash']
        
        size_fit = ['true_to_size']
        
        return {
            'colors': colors,
            'patterns': patterns,
            'style_attributes': style,
            'material_properties': material,
            'occasion_suitability': occasions,
            'seasonal_appropriate': seasons,
            'care_instructions': care,
            'size_fit': size_fit
        }
    
    def _analyze_label_distributions(self, training_samples):
        """Analyze and display label distributions"""
        print(f"\n📊 ANALYZING LABEL DISTRIBUTIONS")
        print("-" * 40)
        
        for attribute_name in self.attributes_config.keys():
            label_counter = Counter()
            label_counts_per_sample = []
            
            for sample in training_samples:
                labels = sample['multilabel_attributes'].get(attribute_name, [])
                label_counts_per_sample.append(len(labels))
                for label in labels:
                    label_counter[label] += 1
            
            print(f"\n{attribute_name.upper()}:")
            print(f"   Average labels per sample: {np.mean(label_counts_per_sample):.2f}")
            print(f"   Max labels per sample: {max(label_counts_per_sample)}")
            print(f"   Most common labels:")
            
            for label, count in label_counter.most_common(5):
                percentage = (count / len(training_samples)) * 100
                print(f"      {label}: {count} ({percentage:.1f}%)")
    
    def prepare_multilabel_training_data(self, training_samples, attribute_name):
        """Prepare training data for a specific attribute"""
        print(f"\n📋 PREPARING TRAINING DATA FOR: {attribute_name}")
        
        # Extract labels for this attribute
        all_labels = []
        for sample in training_samples:
            labels = sample['multilabel_attributes'].get(attribute_name, [])
            all_labels.append(labels)
        
        # Create multi-label binarizer
        mlb = MultiLabelBinarizer()
        y_binary = mlb.fit_transform(all_labels)
        
        print(f"   Samples: {len(all_labels)}")
        print(f"   Classes: {len(mlb.classes_)}")
        print(f"   Label matrix shape: {y_binary.shape}")
        print(f"   Average labels per sample: {np.mean(np.sum(y_binary, axis=1)):.2f}")
        
        # Store the encoder for later use
        self.label_encoders[attribute_name] = mlb
        
        return y_binary, mlb
    
    def create_comprehensive_multilabel_system(self):
        """Create the complete multi-label attribute prediction system"""
        print("🚀 CREATING COMPREHENSIVE MULTI-LABEL ATTRIBUTE SYSTEM")
        print("=" * 70)
        
        # Step 1: Load base model
        if not self.load_base_model_and_create_feature_extractor():
            return False
        
        # Step 2: Generate realistic training data
        training_samples = self.generate_realistic_multilabel_data(num_samples=2000)
        
        # Step 3: Create multi-label models for each attribute
        print(f"\n🏗️ CREATING MULTI-LABEL MODELS FOR {len(self.attributes_config)} ATTRIBUTES")
        
        for attribute_name in self.attributes_config.keys():
            print(f"\n--- Processing {attribute_name} ---")
            
            # Prepare training data
            y_binary, mlb = self.prepare_multilabel_training_data(training_samples, attribute_name)
            
            # Create model
            model = self.create_advanced_multilabel_model(attribute_name)
            self.attribute_models[attribute_name] = {
                'model': model,
                'encoder': mlb,
                'config': self.attributes_config[attribute_name]
            }
        
        print(f"\n✅ ALL MULTI-LABEL MODELS CREATED!")
        
        # Step 4: Save system configuration
        self.save_multilabel_system()
        
        return True
    
    def predict_multilabel_attributes(self, image_path, threshold=0.5):
        """Predict multi-label attributes for an image"""
        print(f"\n🎯 PREDICTING MULTI-LABEL ATTRIBUTES")
        print(f"Image: {image_path}")
        print(f"Threshold: {threshold}")
        
        try:
            # Load and preprocess image
            img = load_img(image_path, target_size=(224, 224))
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Get base category first
            base_prediction = self.base_model.predict(img_array, verbose=0)
            base_category_idx = np.argmax(base_prediction[0])
            base_category = self.base_categories[base_category_idx]
            base_confidence = base_prediction[0][base_category_idx]
            
            print(f"\n📋 BASE CLASSIFICATION:")
            print(f"   Category: {base_category}")
            print(f"   Confidence: {base_confidence:.1%}")
            
            # Predict all attributes
            all_predictions = {}
            
            print(f"\n🏷️ MULTI-LABEL ATTRIBUTE PREDICTIONS:")
            
            for attribute_name, model_info in self.attribute_models.items():
                model = model_info['model']
                encoder = model_info['encoder']
                config = model_info['config']
                
                # Predict probabilities
                probabilities = model.predict(img_array, verbose=0)[0]
                
                # Apply threshold
                predicted_binary = (probabilities > threshold).astype(int)
                
                # Decode to labels
                predicted_labels = encoder.inverse_transform([predicted_binary])
                if len(predicted_labels) > 0 and len(predicted_labels[0]) > 0:
                    labels_with_confidence = []
                    for label in predicted_labels[0]:
                        label_idx = encoder.classes_.tolist().index(label)
                        confidence = probabilities[label_idx]
                        labels_with_confidence.append({
                            'label': label,
                            'confidence': float(confidence)
                        })
                    
                    # Sort by confidence
                    labels_with_confidence.sort(key=lambda x: x['confidence'], reverse=True)
                    all_predictions[attribute_name] = labels_with_confidence
                    
                    print(f"   {attribute_name}:")
                    for item in labels_with_confidence:
                        print(f"      • {item['label']} ({item['confidence']:.1%})")
                else:
                    all_predictions[attribute_name] = []
                    print(f"   {attribute_name}: No predictions above threshold")
            
            return {
                'base_category': base_category,
                'base_confidence': float(base_confidence),
                'multilabel_attributes': all_predictions,
                'prediction_threshold': threshold
            }
            
        except Exception as e:
            print(f"❌ Error predicting attributes: {e}")
            return None
    
    def save_multilabel_system(self):
        """Save the complete multi-label system"""
        print(f"\n💾 SAVING MULTI-LABEL SYSTEM")
        
        # Save configuration
        system_config = {
            'creation_date': datetime.now().isoformat(),
            'base_model': self.base_model_path,
            'attributes_config': self.attributes_config,
            'training_config': self.training_config,
            'total_attribute_models': len(self.attribute_models),
            'system_type': 'Multi-label Attribute Prediction'
        }
        
        with open('multilabel_system_config.json', 'w') as f:
            json.dump(system_config, f, indent=2)
        
        # Save label encoders
        with open('multilabel_encoders.pkl', 'wb') as f:
            pickle.dump(self.label_encoders, f)
        
        print("✅ Multi-label system saved:")
        print("   • multilabel_system_config.json - System configuration")
        print("   • multilabel_training_data.json - Training data")
        print("   • multilabel_encoders.pkl - Label encoders")

def run_step3_development():
    """Run Step 3: Multi-label Attribute Prediction System Development"""
    print("🎯 STEP 3: MULTI-LABEL ATTRIBUTE PREDICTION SYSTEM")
    print("=" * 80)
    print("Goal: Create advanced multi-label attribute classification")
    print("Input: Excellent base model + multi-label training data")
    print("Output: Comprehensive multi-attribute prediction system")
    print("=" * 80)
    
    # Initialize multi-label predictor
    predictor = MultiLabelAttributePredictor()
    
    # Create comprehensive system
    success = predictor.create_comprehensive_multilabel_system()
    
    if success:
        print(f"\n🎉 STEP 3 COMPLETE!")
        print("=" * 40)
        print("✅ Multi-label attribute prediction system created")
        print("✅ 8 different multi-label attribute types")
        print("✅ Realistic training data with label combinations")
        print("✅ Advanced multi-label neural networks")
        print("✅ Binary classification with confidence thresholding")
        
        # Demonstrate the system
        print(f"\n🎯 DEMONSTRATION:")
        print("Multi-label system ready for training and inference")
        
        print(f"\n📁 FILES CREATED:")
        print("   • multilabel_system_config.json - System configuration")
        print("   • multilabel_training_data.json - Multi-label training data")
        print("   • multilabel_encoders.pkl - Label encoders")
        
        print(f"\n🔄 NEXT STEPS FOR STEP 3:")
        print("   1. Collect real multi-label training data")
        print("   2. Train individual multi-label models")
        print("   3. Fine-tune thresholds for each attribute")
        print("   4. Integrate with web application")
        
        print(f"\n➡️ READY FOR STEP 4:")
        print("   Style Compatibility & Recommendation Engine")
        
        return True
    else:
        print("❌ Step 3 development failed")
        return False

if __name__ == "__main__":
    success = run_step3_development()
    
    if success:
        print("\n🚀 Step 3 completed successfully!")
        print("Multi-label attribute prediction system is ready!")
        print("Ready to proceed to Step 4: Style Compatibility Engine")
    else:
        print("\n❌ Step 3 failed - check configuration and try again")
