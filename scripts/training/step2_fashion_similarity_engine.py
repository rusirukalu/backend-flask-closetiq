# step2_fashion_similarity_engine.py - Advanced Fashion Similarity & Retrieval System

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import json
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from collections import defaultdict
import faiss  # Install with: pip install faiss-cpu

class FashionSimilarityEngine:
    """Advanced Fashion Item Similarity & Retrieval Engine"""
    
    def __init__(self, base_model_path='final_enhanced_model.keras'):
        self.base_model_path = base_model_path
        self.base_model = None
        self.feature_extractor = None
        self.similarity_index = None
        self.faiss_index = None
        
        # Wardrobe database
        self.wardrobe_items = []
        self.item_features = {}
        self.item_metadata = {}
        
        # Similarity configuration
        self.similarity_config = {
            'feature_dim': 128,  # Based on your model's feature layer
            'similarity_methods': ['cosine', 'euclidean', 'faiss_l2'],
            'clustering_enabled': True,
            'pca_enabled': True,
            'pca_components': 64
        }
        
        # Base categories from your model
        self.base_categories = [
            'shirts_blouses', 'tshirts_tops', 'dresses', 'pants_jeans',
            'shorts', 'skirts', 'jackets_coats', 'sweaters', 
            'shoes_sneakers', 'shoes_formal', 'bags_accessories'
        ]
    
    def load_feature_extraction_model(self):
        """Load base model and create feature extractor"""
        print("🔧 LOADING FEATURE EXTRACTION MODEL")
        print("=" * 50)
        
        try:
            # Load your excellent base model
            self.base_model = load_model(self.base_model_path)
            print(f"✅ Base model loaded: {self.base_model_path}")
            
            # Create feature extractor (before final classification layer)
            # Extract from the layer before final dense layer for rich features
            feature_layer = self.base_model.layers[-3]  # Adjust based on your model structure
            
            self.feature_extractor = Model(
                inputs=self.base_model.input,
                outputs=feature_layer.output,
                name='fashion_feature_extractor'
            )
            
            print(f"✅ Feature extractor created:")
            print(f"   Input shape: {self.feature_extractor.input_shape}")
            print(f"   Output shape: {self.feature_extractor.output_shape}")
            print(f"   Feature dimensions: {self.feature_extractor.output_shape[-1]}")
            
            # Update feature dimension
            self.similarity_config['feature_dim'] = self.feature_extractor.output_shape[-1]
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading feature extraction model: {e}")
            return False
    
    def extract_image_features(self, image_path):
        """Extract features from a single image"""
        try:
            # Load and preprocess image
            img = load_img(image_path, target_size=(224, 224))
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Extract features
            features = self.feature_extractor.predict(img_array, verbose=0)
            
            # Flatten if needed
            if len(features.shape) > 2:
                features = features.reshape(features.shape[0], -1)
            
            return features[0]  # Return single feature vector
            
        except Exception as e:
            print(f"❌ Error extracting features from {image_path}: {e}")
            return None
    
    def build_wardrobe_feature_database(self, wardrobe_directory='uploads'):
        """Build feature database from existing wardrobe images"""
        print(f"\n📊 BUILDING WARDROBE FEATURE DATABASE")
        print("=" * 50)
        
        if not os.path.exists(wardrobe_directory):
            print(f"❌ Wardrobe directory not found: {wardrobe_directory}")
            return False
        
        # Find all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        for file in os.listdir(wardrobe_directory):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(wardrobe_directory, file))
        
        if not image_files:
            print("❌ No images found in wardrobe directory")
            return False
        
        print(f"📁 Found {len(image_files)} images in wardrobe")
        
        # Extract features for all images
        features_list = []
        valid_items = []
        
        for i, image_path in enumerate(image_files):
            print(f"   Processing {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
            
            # Extract features
            features = self.extract_image_features(image_path)
            
            if features is not None:
                # Get base category prediction
                category_pred = self.predict_base_category(image_path)
                
                # Create item metadata
                item_metadata = {
                    'id': i,
                    'filename': os.path.basename(image_path),
                    'full_path': image_path,
                    'predicted_category': category_pred['category'],
                    'category_confidence': category_pred['confidence'],
                    'extraction_date': datetime.now().isoformat()
                }
                
                features_list.append(features)
                valid_items.append(item_metadata)
                self.item_features[i] = features
                self.item_metadata[i] = item_metadata
        
        self.wardrobe_items = valid_items
        
        if len(features_list) == 0:
            print("❌ No valid features extracted")
            return False
        
        # Convert to numpy array
        self.features_matrix = np.array(features_list)
        
        print(f"✅ Feature database built:")
        print(f"   Valid items: {len(valid_items)}")
        print(f"   Feature matrix shape: {self.features_matrix.shape}")
        
        # Save feature database
        self.save_feature_database()
        
        return True
    
    def predict_base_category(self, image_path):
        """Predict base category for an image"""
        try:
            # Load and preprocess image
            img = load_img(image_path, target_size=(224, 224))
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Predict category
            predictions = self.base_model.predict(img_array, verbose=0)
            predicted_idx = np.argmax(predictions[0])
            confidence = predictions[0][predicted_idx]
            
            return {
                'category': self.base_categories[predicted_idx],
                'confidence': float(confidence),
                'all_predictions': {self.base_categories[i]: float(predictions[0][i]) 
                                  for i in range(len(self.base_categories))}
            }
            
        except Exception as e:
            return {
                'category': 'unknown',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def build_similarity_indexes(self):
        """Build different types of similarity indexes"""
        print(f"\n🔍 BUILDING SIMILARITY INDEXES")
        print("=" * 40)
        
        if self.features_matrix is None:
            print("❌ No feature matrix available")
            return False
        
        # 1. FAISS Index for fast similarity search
        print("   Building FAISS index...")
        try:
            # Normalize features for cosine similarity
            normalized_features = self.features_matrix / np.linalg.norm(self.features_matrix, axis=1, keepdims=True)
            
            # Create FAISS index
            dimension = normalized_features.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner Product for cosine similarity
            self.faiss_index.add(normalized_features.astype('float32'))
            
            print(f"   ✅ FAISS index built with {self.faiss_index.ntotal} items")
            
        except Exception as e:
            print(f"   ⚠️ FAISS index creation failed: {e}")
            self.faiss_index = None
        
        # 2. PCA for dimensionality reduction (optional)
        if self.similarity_config['pca_enabled']:
            print("   Building PCA transformation...")
            try:
                from sklearn.decomposition import PCA
                
                n_components = min(self.similarity_config['pca_components'], 
                                 self.features_matrix.shape[1], 
                                 self.features_matrix.shape[0])
                
                self.pca = PCA(n_components=n_components)
                self.pca_features = self.pca.fit_transform(self.features_matrix)
                
                print(f"   ✅ PCA transformation built: {self.features_matrix.shape[1]} → {n_components} dimensions")
                print(f"   Explained variance ratio: {self.pca.explained_variance_ratio_.sum():.3f}")
                
            except Exception as e:
                print(f"   ⚠️ PCA transformation failed: {e}")
                self.pca = None
        
        # 3. K-Means clustering for style groups
        if self.similarity_config['clustering_enabled']:
            print("   Building style clusters...")
            try:
                # Optimal number of clusters (heuristic: sqrt of items)
                n_clusters = min(int(np.sqrt(len(self.wardrobe_items))), 10, len(self.wardrobe_items))
                n_clusters = max(n_clusters, 2)  # At least 2 clusters
                
                self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                self.cluster_labels = self.kmeans.fit_predict(self.features_matrix)
                
                # Add cluster info to metadata
                for i, item in enumerate(self.wardrobe_items):
                    self.item_metadata[item['id']]['style_cluster'] = int(self.cluster_labels[i])
                
                print(f"   ✅ Style clustering built: {n_clusters} clusters")
                
                # Analyze clusters
                cluster_analysis = defaultdict(list)
                for i, label in enumerate(self.cluster_labels):
                    cluster_analysis[label].append(self.wardrobe_items[i]['predicted_category'])
                
                print("   📊 Cluster analysis:")
                for cluster_id, categories in cluster_analysis.items():
                    category_counts = defaultdict(int)
                    for cat in categories:
                        category_counts[cat] += 1
                    main_category = max(category_counts, key=category_counts.get)
                    print(f"      Cluster {cluster_id}: {len(categories)} items, mainly {main_category}")
                
            except Exception as e:
                print(f"   ⚠️ Clustering failed: {e}")
                self.kmeans = None
        
        return True
    
    def find_similar_items(self, query_image_path, top_k=5, method='cosine', category_filter=None):
        """Find top-k similar items to a query image"""
        print(f"\n🔍 FINDING SIMILAR ITEMS")
        print(f"Query: {query_image_path}")
        print(f"Method: {method}, Top-K: {top_k}")
        
        # Extract features from query image
        query_features = self.extract_image_features(query_image_path)
        if query_features is None:
            return None
        
        # Filter items by category if specified
        candidate_indices = range(len(self.wardrobe_items))
        if category_filter:
            candidate_indices = [i for i, item in enumerate(self.wardrobe_items) 
                               if item['predicted_category'] == category_filter]
            print(f"Filtered to {len(candidate_indices)} items in category: {category_filter}")
        
        if len(candidate_indices) == 0:
            return []
        
        similarities = []
        
        # Calculate similarities based on method
        if method == 'cosine':
            # Cosine similarity
            query_norm = query_features / np.linalg.norm(query_features)
            
            for idx in candidate_indices:
                item_features = self.item_features[idx]
                item_norm = item_features / np.linalg.norm(item_features)
                similarity = np.dot(query_norm, item_norm)
                similarities.append((idx, similarity))
                
        elif method == 'euclidean':
            # Euclidean distance (smaller = more similar)
            for idx in candidate_indices:
                item_features = self.item_features[idx]
                distance = np.linalg.norm(query_features - item_features)
                similarity = 1 / (1 + distance)  # Convert distance to similarity
                similarities.append((idx, similarity))
                
        elif method == 'faiss_l2' and self.faiss_index is not None:
            # FAISS similarity search
            query_norm = query_features / np.linalg.norm(query_features)
            query_norm = query_norm.reshape(1, -1).astype('float32')
            
            scores, indices = self.faiss_index.search(query_norm, min(top_k * 2, len(self.wardrobe_items)))
            
            # Filter by category if needed
            similarities = []
            for score, idx in zip(scores[0], indices[0]):
                if category_filter is None or self.wardrobe_items[idx]['predicted_category'] == category_filter:
                    similarities.append((idx, float(score)))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Get top-k results
        top_results = similarities[:top_k]
        
        # Prepare detailed results
        detailed_results = []
        for idx, similarity_score in top_results:
            item = self.wardrobe_items[idx]
            result = {
                'item_id': idx,
                'similarity_score': similarity_score,
                'filename': item['filename'],
                'category': item['predicted_category'],
                'category_confidence': item['category_confidence'],
                'style_cluster': self.item_metadata[idx].get('style_cluster', -1)
            }
            detailed_results.append(result)
        
        return detailed_results
    
    def get_style_recommendations(self, user_preferences=None, occasion=None, top_k=10):
        """Get style-based recommendations"""
        print(f"\n👗 GENERATING STYLE RECOMMENDATIONS")
        print(f"Preferences: {user_preferences}, Occasion: {occasion}")
        
        recommendations = []
        
        # Strategy 1: Cluster-based recommendations
        if self.kmeans is not None:
            # Find items from different style clusters
            cluster_items = defaultdict(list)
            for i, item in enumerate(self.wardrobe_items):
                cluster_id = self.item_metadata[item['id']].get('style_cluster', -1)
                cluster_items[cluster_id].append((i, item))
            
            # Sample from each cluster
            for cluster_id, items in cluster_items.items():
                # Sort by category confidence
                items.sort(key=lambda x: x[1]['category_confidence'], reverse=True)
                # Take top items from each cluster
                for i, (idx, item) in enumerate(items[:max(1, top_k // len(cluster_items))]):
                    recommendations.append({
                        'item_id': idx,
                        'recommendation_type': 'style_cluster',
                        'cluster_id': cluster_id,
                        'filename': item['filename'],
                        'category': item['predicted_category'],
                        'confidence': item['category_confidence'],
                        'score': item['category_confidence'] - (i * 0.1)  # Slight penalty for lower rank
                    })
        
        # Strategy 2: Category diversity recommendations
        category_counts = defaultdict(int)
        for item in self.wardrobe_items:
            category_counts[item['predicted_category']] += 1
        
        # Recommend from underrepresented categories
        rare_categories = [cat for cat, count in category_counts.items() if count <= 2]
        for item in self.wardrobe_items:
            if item['predicted_category'] in rare_categories:
                recommendations.append({
                    'item_id': item['id'],
                    'recommendation_type': 'category_diversity',
                    'filename': item['filename'],
                    'category': item['predicted_category'],
                    'confidence': item['category_confidence'],
                    'score': item['category_confidence'] + 0.2  # Boost for diversity
                })
        
        # Sort by score and remove duplicates
        seen_items = set()
        unique_recommendations = []
        for rec in sorted(recommendations, key=lambda x: x['score'], reverse=True):
            if rec['item_id'] not in seen_items:
                unique_recommendations.append(rec)
                seen_items.add(rec['item_id'])
        
        return unique_recommendations[:top_k]
    
    def save_feature_database(self):
        """Save the complete feature database"""
        print(f"\n💾 SAVING FEATURE DATABASE")
        
        database = {
            'wardrobe_items': self.wardrobe_items,
            'item_features': {str(k): v.tolist() for k, v in self.item_features.items()},
            'item_metadata': self.item_metadata,
            'features_matrix_shape': self.features_matrix.shape,
            'similarity_config': self.similarity_config,
            'creation_date': datetime.now().isoformat()
        }
        
        # Save as JSON
        with open('fashion_similarity_database.json', 'w') as f:
            json.dump(database, f, indent=2)
        
        # Save features matrix as numpy array
        np.save('features_matrix.npy', self.features_matrix)
        
        # Save FAISS index if available
        if self.faiss_index is not None:
            faiss.write_index(self.faiss_index, 'fashion_similarity.index')
        
        # Save clustering model if available
        if hasattr(self, 'kmeans') and self.kmeans is not None:
            with open('style_clusters.pkl', 'wb') as f:
                pickle.dump(self.kmeans, f)
        
        print("✅ Feature database saved:")
        print("   • fashion_similarity_database.json")
        print("   • features_matrix.npy")
        if self.faiss_index: print("   • fashion_similarity.index")
        if hasattr(self, 'kmeans'): print("   • style_clusters.pkl")
    
    def load_feature_database(self):
        """Load existing feature database"""
        print(f"\n📂 LOADING FEATURE DATABASE")
        
        try:
            # Load main database
            with open('fashion_similarity_database.json', 'r') as f:
                database = json.load(f)
            
            self.wardrobe_items = database['wardrobe_items']
            self.item_features = {int(k): np.array(v) for k, v in database['item_features'].items()}
            self.item_metadata = {int(k): v for k, v in database['item_metadata'].items()}
            self.similarity_config = database['similarity_config']
            
            # Load features matrix
            self.features_matrix = np.load('features_matrix.npy')
            
            # Load FAISS index if available
            if os.path.exists('fashion_similarity.index'):
                self.faiss_index = faiss.read_index('fashion_similarity.index')
            
            # Load clustering model if available
            if os.path.exists('style_clusters.pkl'):
                with open('style_clusters.pkl', 'rb') as f:
                    self.kmeans = pickle.load(f)
            
            print("✅ Feature database loaded:")
            print(f"   Items: {len(self.wardrobe_items)}")
            print(f"   Features matrix: {self.features_matrix.shape}")
            print(f"   FAISS index: {'Available' if self.faiss_index else 'Not available'}")
            print(f"   Clustering: {'Available' if hasattr(self, 'kmeans') else 'Not available'}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading feature database: {e}")
            return False

def run_step2_development():
    """Run Step 2: Fashion Similarity & Retrieval Engine Development"""
    print("🎯 STEP 2: FASHION SIMILARITY & RETRIEVAL ENGINE")
    print("=" * 80)
    print("Goal: Build advanced similarity search and style recommendations")
    print("Input: Excellent base model + wardrobe images")
    print("Output: Comprehensive similarity and retrieval system")
    print("=" * 80)
    
    # Initialize similarity engine
    engine = FashionSimilarityEngine()
    
    # Step 1: Load feature extraction model
    if not engine.load_feature_extraction_model():
        print("❌ Cannot proceed without feature extraction model")
        return False
    
    # Step 2: Build feature database from wardrobe
    success = engine.build_wardrobe_feature_database()
    if not success:
        print("❌ Cannot proceed without feature database")
        return False
    
    # Step 3: Build similarity indexes
    success = engine.build_similarity_indexes()
    if not success:
        print("⚠️ Similarity indexes not fully built, but continuing...")
    
    # Step 4: Demonstrate similarity search
    print(f"\n🎯 DEMONSTRATING SIMILARITY SEARCH")
    print("=" * 40)
    
    if len(engine.wardrobe_items) > 0:
        # Use first item as query example
        query_item = engine.wardrobe_items[0]
        print(f"Query item: {query_item['filename']} ({query_item['predicted_category']})")
        
        # Find similar items
        similar_items = engine.find_similar_items(
            query_item['full_path'], 
            top_k=5, 
            method='cosine'
        )
        
        print(f"\n📊 Top 5 Similar Items:")
        for i, item in enumerate(similar_items, 1):
            print(f"   {i}. {item['filename']} (similarity: {item['similarity_score']:.3f})")
            print(f"      Category: {item['category']} (confidence: {item['category_confidence']:.1%})")
    
    # Step 5: Demonstrate style recommendations
    recommendations = engine.get_style_recommendations(top_k=5)
    
    print(f"\n💡 STYLE RECOMMENDATIONS:")
    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. {rec['filename']} ({rec['recommendation_type']})")
        print(f"      Category: {rec['category']} (score: {rec['score']:.3f})")
    
    print(f"\n🎉 STEP 2 COMPLETE!")
    print("=" * 40)
    print("✅ Fashion similarity engine built")
    print("✅ Feature extraction from base model")
    print("✅ Multiple similarity methods available")
    print("✅ Style clustering and recommendations")
    print("✅ FAISS indexing for fast search")
    
    print(f"\n📁 FILES CREATED:")
    print("   • fashion_similarity_database.json - Complete database")
    print("   • features_matrix.npy - Feature vectors")
    print("   • fashion_similarity.index - FAISS index")
    print("   • style_clusters.pkl - Clustering model")
    
    print(f"\n🔄 INTEGRATION READY:")
    print("   1. Add similarity search to web application")
    print("   2. 'Find Similar Items' button for each wardrobe item")
    print("   3. Style-based recommendations page")
    print("   4. Advanced search by visual similarity")
    
    print(f"\n➡️ READY FOR STEP 3:")
    print("   Multi-label Attribute Prediction (Advanced)")
    
    return True

if __name__ == "__main__":
    success = run_step2_development()
    
    if success:
        print("\n🚀 Step 2 completed successfully!")
        print("Fashion similarity and retrieval engine is ready!")
        print("Ready to proceed to Step 3: Multi-label Attribute Prediction")
    else:
        print("\n❌ Step 2 failed - check configuration and try again")
