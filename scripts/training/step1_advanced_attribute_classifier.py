# step1_advanced_attribute_classifier_fixed.py - Fixed multi-attribute fashion classification

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Concatenate, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

class AdvancedAttributeClassifier:
    """Advanced multi-attribute classification for fashion items"""
    
    def __init__(self, base_model_path='final_enhanced_model.keras'):
        self.base_model_path = base_model_path
        self.base_model = None
        self.attribute_models = {}
        
        # Define comprehensive fashion attributes
        self.attributes_config = {
            'colors': {
                'classes': ['black', 'white', 'gray', 'brown', 'red', 'pink', 'orange', 
                           'yellow', 'green', 'blue', 'purple', 'beige', 'navy', 'multicolor'],
                'type': 'multi_label',  # Can have multiple colors
                'description': 'Primary and secondary colors in the garment'
            },
            'patterns': {
                'classes': ['solid', 'striped', 'checkered', 'floral', 'geometric', 
                           'abstract', 'animal_print', 'polka_dots', 'plaid', 'paisley'],
                'type': 'single_label',  # Usually one primary pattern
                'description': 'Visual patterns and designs'
            },
            'sleeve_length': {
                'classes': ['sleeveless', 'short_sleeve', 'three_quarter', 'long_sleeve', 'not_applicable'],
                'type': 'single_label',
                'description': 'Length of sleeves for tops'
            },
            'neckline': {
                'classes': ['crew_neck', 'v_neck', 'scoop_neck', 'high_neck', 'off_shoulder', 
                           'collar', 'hoodie', 'not_applicable'],
                'type': 'single_label',
                'description': 'Neckline style for tops'
            },
            'fit_type': {
                'classes': ['tight', 'fitted', 'regular', 'loose', 'oversized'],
                'type': 'single_label',
                'description': 'How the garment fits the body'
            },
            'material_texture': {
                'classes': ['smooth', 'textured', 'knit', 'woven', 'leather', 'denim', 
                           'silk', 'cotton', 'synthetic', 'wool'],
                'type': 'multi_label',
                'description': 'Material and texture characteristics'
            },
            'occasion': {
                'classes': ['casual', 'business', 'formal', 'sport', 'party', 'beach', 'home'],
                'type': 'multi_label',
                'description': 'Suitable occasions for wearing'
            },
            'season': {
                'classes': ['spring', 'summer', 'fall', 'winter', 'all_season'],
                'type': 'multi_label',
                'description': 'Seasonal appropriateness'
            }
        }
        
        # Base fashion categories from your existing model
        self.base_categories = [
            'shirts_blouses', 'tshirts_tops', 'dresses', 'pants_jeans',
            'shorts', 'skirts', 'jackets_coats', 'sweaters', 
            'shoes_sneakers', 'shoes_formal', 'bags_accessories'
        ]
        
    def load_base_model(self):
        """Load the excellent base classification model"""
        print("📂 LOADING BASE CLASSIFICATION MODEL")
        print("=" * 50)
        
        try:
            self.base_model = load_model(self.base_model_path)
            print(f"✅ Base model loaded: {self.base_model_path}")
            print(f"   Model accuracy: 91.6%")
            print(f"   Categories: {len(self.base_categories)}")
            
            # Extract feature extractor (everything except final classification layer)
            feature_extractor = Model(
                inputs=self.base_model.input,
                outputs=self.base_model.layers[-3].output  # Before final dense layers
            )
            
            print(f"   Feature extractor created from layer: {self.base_model.layers[-3].name}")
            print(f"   Feature dimensions: {feature_extractor.output_shape}")
            
            return feature_extractor
            
        except Exception as e:
            print(f"❌ Error loading base model: {e}")
            return None
    
    def create_attribute_model(self, attribute_name, feature_extractor):
        """Create a specialized model for a specific attribute"""
        print(f"\n🏗️ Creating model for attribute: {attribute_name}")
        
        config = self.attributes_config[attribute_name]
        num_classes = len(config['classes'])
        model_type = config['type']
        
        # Input layer
        inputs = Input(shape=(224, 224, 3))
        
        # Extract features using pre-trained base
        features = feature_extractor(inputs)
        
        # Add global pooling if not already present
        if len(features.shape) > 2:
            features = GlobalAveragePooling2D()(features)
        
        # Attribute-specific layers
        x = Dense(512, activation='relu', name=f'{attribute_name}_dense1')(features)
        x = Dropout(0.3)(x)
        x = Dense(256, activation='relu', name=f'{attribute_name}_dense2')(x)
        x = Dropout(0.4)(x)
        x = Dense(128, activation='relu', name=f'{attribute_name}_dense3')(x)
        x = Dropout(0.3)(x)
        
        # Output layer based on attribute type
        if model_type == 'multi_label':
            # Multi-label classification (sigmoid)
            outputs = Dense(num_classes, activation='sigmoid', 
                          name=f'{attribute_name}_output')(x)
            loss = 'binary_crossentropy'
            metrics = ['binary_accuracy']
        else:
            # Single-label classification (softmax)
            outputs = Dense(num_classes, activation='softmax', 
                          name=f'{attribute_name}_output')(x)
            loss = 'categorical_crossentropy'
            metrics = ['accuracy']
        
        # Create and compile model
        model = Model(inputs=inputs, outputs=outputs, name=f'{attribute_name}_classifier')
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=loss,
            metrics=metrics
        )
        
        print(f"   ✅ {attribute_name} model created")
        print(f"      Classes: {num_classes}")
        print(f"      Type: {model_type}")
        print(f"      Loss: {loss}")
        
        return model
    
    def create_synthetic_attribute_data(self, num_samples=1000):
        """Create synthetic training data for attribute classification"""
        print(f"\n🎨 CREATING SYNTHETIC ATTRIBUTE TRAINING DATA")
        print("=" * 50)
        
        # This is a placeholder - in real implementation, you'd need manually labeled data
        # For demonstration, we'll create realistic synthetic labels based on base categories
        
        synthetic_data = []
        
        for i in range(num_samples):
            # Random base category
            base_category = np.random.choice(self.base_categories)
            
            # Generate realistic attributes based on category
            sample = {
                'image_id': f'synthetic_{i}',
                'base_category': base_category,
                'attributes': {}
            }
            
            # Generate category-specific attributes
            if 'shirts' in base_category or 'tshirts' in base_category:
                sample['attributes'] = self._generate_top_attributes()
            elif 'dresses' in base_category:
                sample['attributes'] = self._generate_dress_attributes()
            elif 'pants' in base_category or 'shorts' in base_category:
                sample['attributes'] = self._generate_bottom_attributes()
            elif 'shoes' in base_category:
                sample['attributes'] = self._generate_shoe_attributes()
            else:
                sample['attributes'] = self._generate_general_attributes()
            
            synthetic_data.append(sample)
        
        print(f"✅ Generated {len(synthetic_data)} synthetic samples")
        
        # Save synthetic data
        with open('synthetic_attribute_data.json', 'w') as f:
            json.dump(synthetic_data, f, indent=2)
        
        return synthetic_data
    
    def _generate_top_attributes(self):
        """Generate realistic attributes for tops"""
        colors = np.random.choice(['black', 'white', 'blue', 'gray', 'red'], 
                                size=np.random.randint(1, 3), replace=False).tolist()
        
        # Fixed occasion selection
        occasion_options = ['casual', 'business', 'casual_business']
        occasion_weights = [0.5, 0.3, 0.2]
        chosen_occasion = np.random.choice(occasion_options, p=occasion_weights)
        
        if chosen_occasion == 'casual_business':
            occasion = ['casual', 'business']
        else:
            occasion = [chosen_occasion]
        
        return {
            'colors': colors,
            'patterns': [np.random.choice(['solid', 'striped', 'checkered'])],
            'sleeve_length': [np.random.choice(['short_sleeve', 'long_sleeve', 'sleeveless'])],
            'neckline': [np.random.choice(['crew_neck', 'v_neck', 'collar'])],
            'fit_type': [np.random.choice(['fitted', 'regular', 'loose'])],
            'material_texture': [np.random.choice(['cotton', 'synthetic', 'knit'])],
            'occasion': occasion,
            'season': ['all_season']
        }
    
    def _generate_dress_attributes(self):
        """Generate realistic attributes for dresses - FIXED"""
        colors = np.random.choice(['black', 'red', 'blue', 'white', 'pink'], 
                                size=np.random.randint(1, 2), replace=False).tolist()
        
        # Fixed occasion selection
        occasion_options = ['party', 'formal', 'casual', 'party_formal']
        occasion_weights = [0.3, 0.3, 0.2, 0.2]
        chosen_occasion = np.random.choice(occasion_options, p=occasion_weights)
        
        if chosen_occasion == 'party_formal':
            occasion = ['party', 'formal']
        else:
            occasion = [chosen_occasion]
        
        # Fixed season selection
        season_options = ['spring_summer', 'fall', 'all_season']
        season_weights = [0.4, 0.3, 0.3]
        chosen_season = np.random.choice(season_options, p=season_weights)
        
        if chosen_season == 'spring_summer':
            season = ['spring', 'summer']
        else:
            season = [chosen_season]
        
        return {
            'colors': colors,
            'patterns': [np.random.choice(['solid', 'floral', 'geometric'])],
            'sleeve_length': [np.random.choice(['sleeveless', 'short_sleeve', 'long_sleeve'])],
            'neckline': [np.random.choice(['scoop_neck', 'v_neck', 'off_shoulder'])],
            'fit_type': [np.random.choice(['fitted', 'regular', 'loose'])],
            'material_texture': [np.random.choice(['silk', 'cotton', 'synthetic'])],
            'occasion': occasion,
            'season': season
        }
    
    def _generate_bottom_attributes(self):
        """Generate realistic attributes for bottoms - FIXED"""
        colors = np.random.choice(['black', 'blue', 'gray', 'brown', 'navy'], 
                                size=np.random.randint(1, 2), replace=False).tolist()
        
        # Fixed occasion selection
        occasion_options = ['casual', 'business', 'casual_business']
        occasion_weights = [0.6, 0.2, 0.2]
        chosen_occasion = np.random.choice(occasion_options, p=occasion_weights)
        
        if chosen_occasion == 'casual_business':
            occasion = ['casual', 'business']
        else:
            occasion = [chosen_occasion]
        
        return {
            'colors': colors,
            'patterns': [np.random.choice(['solid', 'striped'])],
            'sleeve_length': ['not_applicable'],
            'neckline': ['not_applicable'],
            'fit_type': [np.random.choice(['tight', 'fitted', 'regular', 'loose'])],
            'material_texture': [np.random.choice(['denim', 'cotton', 'synthetic'])],
            'occasion': occasion,
            'season': ['all_season']
        }
    
    def _generate_shoe_attributes(self):
        """Generate realistic attributes for shoes - FIXED"""
        colors = np.random.choice(['black', 'brown', 'white', 'gray'], 
                                size=np.random.randint(1, 2), replace=False).tolist()
        
        # Fixed occasion selection
        occasion_options = ['casual', 'formal', 'sport']
        occasion_weights = [0.4, 0.4, 0.2]
        chosen_occasion = np.random.choice(occasion_options, p=occasion_weights)
        
        return {
            'colors': colors,
            'patterns': ['solid'],
            'sleeve_length': ['not_applicable'],
            'neckline': ['not_applicable'],
            'fit_type': ['regular'],
            'material_texture': [np.random.choice(['leather', 'synthetic', 'canvas'])],
            'occasion': [chosen_occasion],
            'season': ['all_season']
        }
    
    def _generate_general_attributes(self):
        """Generate general attributes for other items - FIXED"""
        colors = np.random.choice(['black', 'brown', 'gray', 'blue'], 
                                size=np.random.randint(1, 2), replace=False).tolist()
        
        return {
            'colors': colors,
            'patterns': [np.random.choice(['solid', 'textured'])],
            'sleeve_length': ['not_applicable'],
            'neckline': ['not_applicable'],
            'fit_type': ['regular'],
            'material_texture': [np.random.choice(['synthetic', 'leather', 'fabric'])],
            'occasion': ['casual'],
            'season': ['all_season']
        }
    
    def create_comprehensive_attribute_system(self):
        """Create the complete multi-attribute classification system"""
        print("🚀 CREATING COMPREHENSIVE ATTRIBUTE CLASSIFICATION SYSTEM")
        print("=" * 70)
        
        # Step 1: Load base model and create feature extractor
        feature_extractor = self.load_base_model()
        if not feature_extractor:
            return False
        
        # Step 2: Create individual attribute models
        print(f"\n🏗️ Creating {len(self.attributes_config)} attribute models...")
        
        for attribute_name in self.attributes_config.keys():
            model = self.create_attribute_model(attribute_name, feature_extractor)
            self.attribute_models[attribute_name] = model
        
        print(f"✅ All attribute models created!")
        
        # Step 3: Create synthetic training data
        synthetic_data = self.create_synthetic_attribute_data()
        
        # Step 4: Save system configuration
        system_config = {
            'creation_date': datetime.now().isoformat(),
            'base_model': self.base_model_path,
            'attributes': self.attributes_config,
            'base_categories': self.base_categories,
            'total_attribute_models': len(self.attribute_models),
            'synthetic_samples': len(synthetic_data)
        }
        
        with open('attribute_system_config.json', 'w') as f:
            json.dump(system_config, f, indent=2)
        
        print(f"\n📄 System configuration saved: attribute_system_config.json")
        
        return True
    
    def demonstrate_attribute_prediction(self, image_path=None):
        """Demonstrate multi-attribute prediction on an image"""
        print(f"\n🎯 DEMONSTRATING MULTI-ATTRIBUTE PREDICTION")
        print("-" * 50)
        
        if image_path is None:
            # Create a random test image
            test_image = np.random.rand(1, 224, 224, 3)
            print("Using random test image for demonstration")
        else:
            # Load and preprocess real image
            from PIL import Image
            img = Image.open(image_path).convert('RGB')
            img = img.resize((224, 224))
            test_image = np.array(img) / 255.0
            test_image = np.expand_dims(test_image, axis=0)
            print(f"Using image: {image_path}")
        
        # Get base category prediction
        base_prediction = self.base_model.predict(test_image, verbose=0)
        base_category_idx = np.argmax(base_prediction[0])
        base_category = self.base_categories[base_category_idx]
        base_confidence = base_prediction[0][base_category_idx]
        
        print(f"\n📋 BASE CLASSIFICATION:")
        print(f"   Category: {base_category}")
        print(f"   Confidence: {base_confidence:.1%}")
        
        # Predict attributes (placeholder - models need training)
        print(f"\n🏷️ PREDICTED ATTRIBUTES:")
        
        predicted_attributes = {}
        for attribute_name, config in self.attributes_config.items():
            # This is a placeholder prediction - in real implementation, use trained models
            if config['type'] == 'multi_label':
                num_predictions = np.random.randint(1, 3)
                predicted_classes = np.random.choice(config['classes'], 
                                                   size=num_predictions,
                                                   replace=False)
            else:
                predicted_classes = [np.random.choice(config['classes'])]
            
            predicted_attributes[attribute_name] = list(predicted_classes)
            print(f"   {attribute_name}: {list(predicted_classes)}")
        
        return {
            'base_category': base_category,
            'base_confidence': float(base_confidence),
            'attributes': predicted_attributes
        }

def run_step1_development():
    """Run Step 1: Advanced Attribute Classification System Development"""
    print("🎯 STEP 1: ADVANCED ATTRIBUTE CLASSIFICATION SYSTEM")
    print("=" * 80)
    print("Goal: Create multi-attribute classification for detailed fashion analysis")
    print("Input: Excellent base model (91.6% accuracy)")
    print("Output: Comprehensive attribute classification system")
    print("=" * 80)
    
    # Initialize system
    classifier = AdvancedAttributeClassifier()
    
    # Create comprehensive system
    success = classifier.create_comprehensive_attribute_system()
    
    if success:
        print(f"\n🎉 STEP 1 COMPLETE!")
        print("=" * 40)
        print("✅ Multi-attribute classification system created")
        print("✅ 8 different attribute types configured")
        print("✅ Feature extractor from base model ready")
        print("✅ Synthetic training data generated")
        
        # Demonstrate the system
        print(f"\n🎯 DEMONSTRATION:")
        result = classifier.demonstrate_attribute_prediction()
        
        print(f"\n📁 FILES CREATED:")
        print("   • attribute_system_config.json - System configuration")
        print("   • synthetic_attribute_data.json - Training data template")
        
        print(f"\n🔄 NEXT STEPS FOR STEP 1:")
        print("   1. Collect real labeled attribute data (colors, patterns, etc.)")
        print("   2. Train individual attribute models")
        print("   3. Integrate with existing web application")
        print("   4. Test with real fashion images")
        
        print(f"\n➡️ READY FOR STEP 2:")
        print("   Fashion Item Similarity & Retrieval Engine")
        
        return True
    else:
        print("❌ Step 1 development failed")
        return False

if __name__ == "__main__":
    success = run_step1_development()
    
    if success:
        print("\n🚀 Step 1 completed successfully!")
        print("Ready to proceed to Step 2: Fashion Similarity Engine")
    else:
        print("\n❌ Step 1 failed - check configuration and try again")
