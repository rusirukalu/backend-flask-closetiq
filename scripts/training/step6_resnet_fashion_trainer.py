# step6_resnet_fashion_trainer.py - Train ResNet-50 for Real Fashion

import os
import tensorflow as tf
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import *
import numpy as np

class ResNetFashionClassifier:
    def __init__(self, num_classes=11, img_size=224):
        self.num_classes = num_classes
        self.img_size = img_size
        self.model = None
        
        # Categories from research papers
        self.categories = [
            'shirts_blouses', 'tshirts_tops', 'dresses', 'pants_jeans', 
            'shorts', 'skirts', 'jackets_coats', 'sweaters', 
            'shoes_sneakers', 'shoes_formal', 'bags_accessories'
        ]
    
    def create_research_based_model(self):
        """Create ResNet-50 based model as per research papers"""
        print("🏗️ Building ResNet-50 Fashion Classifier (Research-Based)...")
        
        # ResNet-50 backbone (as mentioned in Paper 4)
        base_model = ResNet50V2(
            weights='imagenet',
            include_top=False,
            input_shape=(self.img_size, self.img_size, 3),
            pooling=None
        )
        
        # Freeze initial layers, fine-tune later layers
        for layer in base_model.layers[:-20]:
            layer.trainable = False
        
        # Input
        inputs = Input(shape=(self.img_size, self.img_size, 3))
        
        # Data augmentation (built-in)
        x = RandomFlip("horizontal")(inputs)
        x = RandomRotation(0.1)(x)
        x = RandomZoom(0.1)(x)
        
        # ResNet backbone
        x = base_model(x, training=False)
        
        # Research-based classification head
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        # Multi-scale feature extraction (from Paper 3)
        branch1 = Dense(512, activation='relu')(x)
        branch1 = BatchNormalization()(branch1)
        branch1 = Dropout(0.4)(branch1)
        
        branch2 = Dense(256, activation='relu')(x)
        branch2 = BatchNormalization()(branch2)
        branch2 = Dropout(0.3)(branch2)
        
        # Combine branches
        combined = concatenate([branch1, branch2])
        
        # Final classification
        x = Dense(256, activation='relu')(combined)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        
        outputs = Dense(self.num_classes, activation='softmax')(x)
        
        # Create model
        self.model = Model(inputs=inputs, outputs=outputs)
        
        # Compile with research-based settings
        self.model.compile(
            optimizer=Adam(learning_rate=1e-4),  # Lower LR for transfer learning
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_3_categorical_accuracy']
        )
        
        print("✅ Research-based ResNet-50 model created!")
        return self.model
    
    def setup_real_data_generators(self, dataset_path):
        """Setup data generators for real fashion images"""
        print("📊 Setting up real fashion data generators...")
        
        # Heavy augmentation for real images (as per research)
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=30,
            width_shift_range=0.3,
            height_shift_range=0.3,
            shear_range=0.2,
            zoom_range=0.3,
            horizontal_flip=True,
            brightness_range=[0.7, 1.3],
            channel_shift_range=0.2,
            fill_mode='nearest'
        )
        
        val_test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create generators
        train_generator = train_datagen.flow_from_directory(
            os.path.join(dataset_path, 'train'),
            target_size=(self.img_size, self.img_size),
            batch_size=32,
            class_mode='categorical',
            shuffle=True,
            classes=self.categories
        )
        
        val_generator = val_test_datagen.flow_from_directory(
            os.path.join(dataset_path, 'validation'),
            target_size=(self.img_size, self.img_size),
            batch_size=32,
            class_mode='categorical',
            shuffle=False,
            classes=self.categories
        )
        
        test_generator = val_test_datagen.flow_from_directory(
            os.path.join(dataset_path, 'test'),
            target_size=(self.img_size, self.img_size),
            batch_size=32,
            class_mode='categorical',
            shuffle=False,
            classes=self.categories
        )
        
        return train_generator, val_generator, test_generator
    
    def train_research_model(self, dataset_path, epochs=50):
        """Train model using research-based methodology"""
        print("🚀 Starting Research-Based Training...")
        
        # Setup data
        train_gen, val_gen, test_gen = self.setup_real_data_generators(dataset_path)
        
        # Research-based callbacks
        callbacks = [
            ModelCheckpoint(
                'resnet_fashion_classifier.h5',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            # Learning rate scheduling (from research papers)
            LearningRateScheduler(
                lambda epoch: 1e-4 * 0.9 ** epoch
            )
        ]
        
        # Phase 1: Feature extraction (freeze backbone)
        print("Phase 1: Feature extraction training...")
        history1 = self.model.fit(
            train_gen,
            epochs=epochs//2,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )
        
        # Phase 2: Fine-tuning (unfreeze backbone)
        print("Phase 2: Fine-tuning full model...")
        
        # Unfreeze more layers
        for layer in self.model.layers[-30:]:
            if hasattr(layer, 'layers'):  # If it's a nested model
                for sublayer in layer.layers:
                    sublayer.trainable = True
            else:
                layer.trainable = True
        
        # Recompile with lower learning rate
        self.model.compile(
            optimizer=Adam(learning_rate=1e-5),  # Much lower for fine-tuning
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_3_categorical_accuracy']
        )
        
        # Continue training
        history2 = self.model.fit(
            train_gen,
            epochs=epochs,
            initial_epoch=epochs//2,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )
        
        # Final evaluation
        print("📊 Final evaluation on test set...")
        test_results = self.model.evaluate(test_gen, verbose=1)
        
        print(f"✅ Research-Based Training Complete!")
        print(f"Final Test Accuracy: {test_results[1]:.4f}")
        print(f"Final Top-3 Accuracy: {test_results[2]:.4f}")
        
        return history1, history2, test_results

def count_images_in_dataset(dataset_path):
    """Count total images in dataset and provide detailed breakdown"""
    total_images = 0
    split_counts = {'train': 0, 'validation': 0, 'test': 0}
    category_counts = {}
    
    if not os.path.exists(dataset_path):
        return 0, split_counts, category_counts
    
    # Define valid image extensions
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp')
    
    for split in ['train', 'validation', 'test']:
        split_path = os.path.join(dataset_path, split)
        if os.path.exists(split_path):
            split_total = 0
            for category in os.listdir(split_path):
                category_path = os.path.join(split_path, category)
                if os.path.isdir(category_path):
                    # Count image files
                    images = [f for f in os.listdir(category_path) 
                             if f.lower().endswith(valid_extensions)]
                    image_count = len(images)
                    split_total += image_count
                    
                    # Track category counts
                    if category not in category_counts:
                        category_counts[category] = {'train': 0, 'validation': 0, 'test': 0}
                    category_counts[category][split] = image_count
            
            split_counts[split] = split_total
            total_images += split_total
    
    return total_images, split_counts, category_counts

def print_dataset_status(dataset_path, categories):
    """Print detailed dataset status and requirements"""
    total_images, split_counts, category_counts = count_images_in_dataset(dataset_path)
    
    print(f"\n📊 DATASET STATUS REPORT")
    print("=" * 50)
    print(f"Total images found: {total_images}")
    print(f"Train: {split_counts['train']} | Validation: {split_counts['validation']} | Test: {split_counts['test']}")
    
    if total_images > 0:
        print(f"\n📁 Images per category:")
        for category in categories:
            if category in category_counts:
                counts = category_counts[category]
                total_cat = sum(counts.values())
                print(f"  {category}: {total_cat} total (Train: {counts['train']}, Val: {counts['validation']}, Test: {counts['test']})")
            else:
                print(f"  {category}: 0 total (missing)")
    
    print(f"\n📋 REQUIREMENTS:")
    print(f"  • Minimum recommended: 2,200 total images (200 per category)")
    print(f"  • Minimum for testing: 550 total images (50 per category)")
    print(f"  • Current status: {'✅ Ready for training' if total_images >= 550 else '❌ Need more images'}")
    
    if total_images < 550:
        print(f"\n💡 SUGGESTIONS:")
        print(f"  • Add {550 - total_images} more images to reach minimum")
        print(f"  • Use web scraping tools for fashion images")
        print(f"  • Download from fashion e-commerce sites")
        print(f"  • Ensure balanced distribution across categories")

# Execute Research-Based Training
def run_research_based_training():
    """Run complete research-based fashion training"""
    
    print("🎯 RESEARCH-BASED FASHION CLASSIFIER TRAINING")
    print("=" * 60)
    print("Following methodology from academic papers...")
    
    # Create classifier
    classifier = ResNetFashionClassifier(num_classes=11)
    
    # Create model
    model = classifier.create_research_based_model()
    
    # Dataset path
    dataset_path = "real_fashion_dataset"
    
    print("\n📋 NEXT STEPS:")
    print("1. Populate 'real_fashion_dataset' with 200+ real images per category")
    print("2. Use sources: Google Images API, Fashion websites, E-commerce sites")
    print("3. Ensure high-quality, diverse clothing images")
    print("4. Run the training script")
    
    # Improved dataset validation
    total_images, split_counts, category_counts = count_images_in_dataset(dataset_path)
    
    # Print detailed dataset status
    print_dataset_status(dataset_path, classifier.categories)
    
    # Check if we have sufficient images for training
    if total_images >= 550:  # Minimum viable dataset size
        print(f"\n🚀 Dataset validation passed! Found {total_images} images. Starting training...")
        try:
            h1, h2, results = classifier.train_research_model(dataset_path)
            return classifier.model, results
        except Exception as e:
            print(f"\n❌ Training failed: {str(e)}")
            print("This might be due to:")
            print("  • Images in wrong format")
            print("  • Corrupted image files")
            print("  • Incorrect directory structure")
            return model, None
    else:
        print(f"\n⚠️ Insufficient images found ({total_images} images).")
        print("Expected directory structure:")
        print("\nreal_fashion_dataset/")
        print("├── train/")
        for category in classifier.categories:
            print(f"│   ├── {category}/")
            print(f"│   │   ├── image1.jpg")
            print(f"│   │   ├── image2.jpg")
            print(f"│   │   └── ...")
        print("├── validation/")
        for category in classifier.categories:
            print(f"│   ├── {category}/")
        print("└── test/")
        for category in classifier.categories:
            print(f"    ├── {category}/")
        
        print(f"\n🔧 Quick Setup Command:")
        print("mkdir -p real_fashion_dataset/{train,validation,test}/{" + ",".join(classifier.categories) + "}")
        
        print("Model architecture created but not trained.")
        return model, None

# Run the training
if __name__ == "__main__":
    model, results = run_research_based_training()
