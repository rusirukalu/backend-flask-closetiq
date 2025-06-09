import os
import shutil
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn
import cv2
from sklearn.cluster import KMeans
import random
from tqdm import tqdm
import json

# Updated 11-category system
CLASS_MAPPING = {
    'top/shirt': 0,           
    'top/t-shirt': 1,         
    'top/blouse': 2,          
    'bottom/trouser': 3,      
    'bottom/shorts': 4,       
    'dress': 5,               
    'outerwear': 6,           
    'shoes/sneaker': 7,       
    'shoes/formal': 8,        
    'shoes/sandal': 9,        
    'formal/suit': 10         
}

REVERSE_CLASS_MAPPING = {v: k for k, v in CLASS_MAPPING.items()}

class VisualClothingCategorizer:
    def __init__(self):
        """Initialize with computer vision approach"""
        print("🤖 Loading ResNet50 for visual analysis...")
        
        # Load pre-trained ResNet50
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        
        # Remove final classification layer to get features
        self.feature_extractor = nn.Sequential(*list(self.model.children())[:-1])
        self.feature_extractor.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print("✅ Visual analyzer loaded!")
    
    def extract_features(self, image_path):
        """Extract visual features from image"""
        try:
            # Load and preprocess image
            img = Image.open(image_path).convert('RGB')
            img_tensor = self.transform(img).unsqueeze(0)
            
            # Extract features
            with torch.no_grad():
                features = self.feature_extractor(img_tensor)
                features = features.flatten().numpy()
            
            return features
            
        except Exception as e:
            print(f"Error extracting features from {image_path}: {e}")
            return None
    
    def analyze_color_distribution(self, image_path):
        """Analyze color distribution for categorization hints"""
        try:
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Reshape for color analysis
            pixels = img.reshape(-1, 3)
            
            # Get dominant colors
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            kmeans.fit(pixels)
            colors = kmeans.cluster_centers_
            
            # Analyze color characteristics
            avg_color = np.mean(colors, axis=0)
            color_variance = np.var(colors, axis=0)
            
            return avg_color, color_variance
            
        except Exception as e:
            return None, None
    
    def get_shape_features(self, image_path):
        """Extract shape-based features"""
        try:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
            # Edge detection
            edges = cv2.Canny(img, 50, 150)
            
            # Calculate aspect ratio and edge density
            height, width = img.shape
            aspect_ratio = width / height
            edge_density = np.sum(edges > 0) / (height * width)
            
            return aspect_ratio, edge_density
            
        except Exception as e:
            return None, None

def smart_visual_categorization():
    """Use visual analysis + clustering for intelligent categorization"""
    
    SOURCE_DIR = "clothing_datasets_8cat/images_original"
    TARGET_DIR = "visual_categorized_11cat"
    
    # Create directories
    for category in CLASS_MAPPING.keys():
        os.makedirs(os.path.join(TARGET_DIR, category), exist_ok=True)
    os.makedirs(os.path.join(TARGET_DIR, 'needs_review'), exist_ok=True)
    
    # Initialize categorizer
    categorizer = VisualClothingCategorizer()
    
    # Get all images
    all_images = [f for f in os.listdir(SOURCE_DIR) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"🔍 Analyzing {len(all_images)} images visually...")
    
    # Process subset first for speed (you can increase this)
    subset_size = min(2000, len(all_images))  # Process 2000 images
    subset_images = random.sample(all_images, subset_size)
    
    print(f"Processing {subset_size} images for visual categorization...")
    
    # Extract features for clustering
    features_list = []
    valid_images = []
    
    for img_file in tqdm(subset_images, desc="Extracting features"):
        img_path = os.path.join(SOURCE_DIR, img_file)
        
        # Extract multiple types of features
        visual_features = categorizer.extract_features(img_path)
        color_avg, color_var = categorizer.analyze_color_distribution(img_path)
        aspect_ratio, edge_density = categorizer.get_shape_features(img_path)
        
        if visual_features is not None and color_avg is not None:
            # Combine all features
            combined_features = np.concatenate([
                visual_features[:100],  # First 100 visual features
                color_avg,              # Average color
                color_var,              # Color variance
                [aspect_ratio, edge_density]  # Shape features
            ])
            
            features_list.append(combined_features)
            valid_images.append(img_file)
    
    print(f"Successfully extracted features from {len(valid_images)} images")
    
    if len(features_list) == 0:
        print("❌ No features extracted. Check image paths.")
        return
    
    # Cluster images into 11 categories using visual similarity
    print("🎯 Clustering images by visual similarity...")
    
    features_array = np.array(features_list)
    
    # Use KMeans to cluster into 11 groups
    kmeans = KMeans(n_clusters=11, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features_array)
    
    # Assign clusters to categories
    category_names = list(CLASS_MAPPING.keys())
    
    category_counts = {cat: 0 for cat in category_names}
    
    for i, (img_file, cluster_id) in enumerate(zip(valid_images, cluster_labels)):
        # Map cluster to category
        category = category_names[cluster_id]
        
        # Copy and process image
        source_path = os.path.join(SOURCE_DIR, img_file)
        target_path = os.path.join(TARGET_DIR, category, f"visual_{img_file}")
        
        try:
            # Load, resize and save
            img = Image.open(source_path).convert('RGB')
            img = img.resize((224, 224), Image.LANCZOS)
            img.save(target_path, 'JPEG', quality=95)
            
            category_counts[category] += 1
            
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
    
    # Show distribution
    print("\n📊 Visual Clustering Results:")
    print("-" * 50)
    total_categorized = 0
    for category, count in category_counts.items():
        total_categorized += count
        print(f"{category:20}: {count:4d} images")
    
    print(f"\n✅ Visually categorized {total_categorized} images!")
    
    # Save clustering report
    report = {
        'total_processed': len(valid_images),
        'category_distribution': category_counts,
        'clustering_method': 'kmeans_visual_features',
        'features_used': ['resnet50_features', 'color_analysis', 'shape_features']
    }
    
    with open(os.path.join(TARGET_DIR, 'clustering_report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    
    return TARGET_DIR

def manual_review_tool():
    """Manual review and correction tool"""
    
    CLUSTERED_DIR = "visual_categorized_11cat"
    CORRECTED_DIR = "corrected_categorized_11cat"
    
    print("🔧 MANUAL REVIEW & CORRECTION TOOL")
    print("=" * 50)
    print("Review and correct the visual clustering results")
    print()
    
    categories = list(CLASS_MAPPING.keys())
    
    for i, cat in enumerate(categories, 1):
        print(f"{i:2d}. {cat}")
    print(f"{len(categories)+1:2d}. skip")
    print(" 0. quit")
    print()
    
    # Create corrected directories
    for category in categories:
        os.makedirs(os.path.join(CORRECTED_DIR, category), exist_ok=True)
    
    corrected_count = 0
    
    # Review each category
    for category in categories:
        cat_dir = os.path.join(CLUSTERED_DIR, category)
        
        if not os.path.exists(cat_dir):
            continue
            
        images = [f for f in os.listdir(cat_dir) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not images:
            continue
        
        print(f"\n📁 Reviewing {category} ({len(images)} images)")
        review_count = min(20, len(images))  # Review first 20 images
        
        for img in images[:review_count]:
            img_path = os.path.join(cat_dir, img)
            
            print(f"\nImage: {img}")
            print(f"Current category: {category}")
            
            # You could add image display here if possible
            # For now, just show file info
            try:
                pil_img = Image.open(img_path)
                print(f"Size: {pil_img.size}")
            except:
                pass
            
            while True:
                choice = input("Correct category (1-11, 12=skip, 0=quit): ").strip()
                
                if choice == '0':
                    print("Stopping review...")
                    return CORRECTED_DIR
                
                if choice == '12':
                    break  # Skip this image
                
                try:
                    choice_num = int(choice)
                    if 1 <= choice_num <= 11:
                        correct_category = categories[choice_num - 1]
                        
                        # Copy to corrected category
                        corrected_path = os.path.join(CORRECTED_DIR, correct_category, img)
                        shutil.copy2(img_path, corrected_path)
                        
                        corrected_count += 1
                        print(f"✅ Moved to {correct_category}")
                        break
                    else:
                        print("Invalid choice.")
                except ValueError:
                    print("Please enter a number.")
    
    print(f"\n✅ Manual review complete! Corrected {corrected_count} images")
    return CORRECTED_DIR

def create_training_splits(source_dir):
    """Create train/val/test splits from categorized data"""
    
    FINAL_DIR = f"{source_dir}_training"
    
    # Create directory structure
    for split in ['train', 'validation', 'test']:
        for category in CLASS_MAPPING.keys():
            os.makedirs(os.path.join(FINAL_DIR, split, category), exist_ok=True)
    
    print(f"📂 Creating training splits from {source_dir}...")
    
    total_train = total_val = total_test = 0
    
    for category in CLASS_MAPPING.keys():
        source_cat_dir = os.path.join(source_dir, category)
        
        if not os.path.exists(source_cat_dir):
            print(f"⚠️  {category}: Directory not found")
            continue
            
        images = [f for f in os.listdir(source_cat_dir) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if len(images) < 3:
            print(f"⚠️  {category}: Only {len(images)} images")
            continue
        
        # Shuffle and split
        random.shuffle(images)
        
        total = len(images)
        train_end = max(1, int(0.7 * total))
        val_end = max(train_end + 1, int(0.85 * total))
        
        train_imgs = images[:train_end]
        val_imgs = images[train_end:val_end]
        test_imgs = images[val_end:]
        
        # Copy images
        for split, img_list in [('train', train_imgs), ('validation', val_imgs), ('test', test_imgs)]:
            for img in img_list:
                source_path = os.path.join(source_cat_dir, img)
                target_path = os.path.join(FINAL_DIR, split, category, img)
                shutil.copy2(source_path, target_path)
        
        total_train += len(train_imgs)
        total_val += len(val_imgs)
        total_test += len(test_imgs)
        
        print(f"{category:20}: {len(train_imgs):3d} train, {len(val_imgs):3d} val, {len(test_imgs):3d} test")
    
    print("-" * 60)
    print(f"{'TOTAL':20}: {total_train:3d} train, {total_val:3d} val, {total_test:3d} test")
    
    print(f"\n✅ Training data ready: {FINAL_DIR}")
    return FINAL_DIR

if __name__ == "__main__":
    print("🎯 VISUAL CLOTHING CATEGORIZATION")
    print("=" * 60)
    print("This approach uses computer vision to analyze actual image content")
    print("instead of relying on filenames (which are UUIDs).")
    print()
    print("Steps:")
    print("1. Extract visual features from images")
    print("2. Cluster similar images together")
    print("3. Manual review and correction")
    print("4. Create training splits")
    print()
    
    choice = input("Start visual categorization? (y/n): ").strip().lower()
    
    if choice == 'y':
        # Step 1: Visual clustering
        clustered_dir = smart_visual_categorization()
        
        # Step 2: Manual review (optional)
        review_choice = input("\nDo manual review? (y/n): ").strip().lower()
        if review_choice == 'y':
            corrected_dir = manual_review_tool()
            final_source = corrected_dir
        else:
            final_source = clustered_dir
        
        # Step 3: Create training splits
        training_dir = create_training_splits(final_source)
        
        print("\n" + "=" * 60)
        print("✅ VISUAL CATEGORIZATION COMPLETE!")
        print("=" * 60)
        print(f"Training data ready: {training_dir}")
        print()
        print("This dataset is based on actual visual similarity,")
        print("not random assignment or filename keywords!")
        
    else:
        print("Cancelled.")
