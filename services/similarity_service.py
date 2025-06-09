# services/similarity_service.py - Enhanced Similarity Search Engine
import numpy as np
import json
import os
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import tensorflow as tf
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class SimilarityEngine:
    """Enhanced Fashion Similarity Search Engine"""
    
    def __init__(self, database_path):
        self.database_path = database_path
        self.feature_database = {}
        self.metadata_database = {}
        self.similarity_matrix = None
        self.feature_extractor = None
        
        # Load existing database
        self.load_database()
        
        # Initialize feature extractor
        self.setup_feature_extractor()
    
    def load_database(self):
        """Load similarity database"""
        try:
            if os.path.exists(self.database_path):
                with open(self.database_path, 'r') as f:
                    data = json.load(f)
                    
                self.feature_database = data.get('features', {})
                self.metadata_database = data.get('metadata', {})
                
                # Convert features back to numpy arrays
                for item_id, features in self.feature_database.items():
                    self.feature_database[item_id] = np.array(features)
                
                logger.info(f"✅ Similarity database loaded: {len(self.feature_database)} items")
                
                # Build similarity matrix
                self.build_similarity_matrix()
                
                return True
            else:
                logger.warning(f"⚠️ Similarity database not found: {self.database_path}")
                return False
        except Exception as e:
            logger.error(f"❌ Error loading similarity database: {str(e)}")
            return False
    
    def is_loaded(self):
        """Check if similarity engine is loaded"""
        return len(self.feature_database) > 0
    
    def setup_feature_extractor(self):
        """Setup feature extractor for new images"""
        try:
            # Use a pre-trained model for feature extraction
            base_model = tf.keras.applications.EfficientNetV2B0(
                weights='imagenet',
                include_top=False,
                pooling='avg'
            )
            
            # Add global average pooling
            self.feature_extractor = tf.keras.models.Model(
                inputs=base_model.input,
                outputs=base_model.output
            )
            
            logger.info("✅ Feature extractor initialized")
            
        except Exception as e:
            logger.error(f"❌ Error setting up feature extractor: {str(e)}")
    
    def extract_features(self, image):
        """Extract features from an image"""
        try:
            if self.feature_extractor is None:
                raise Exception("Feature extractor not initialized")
            
            # Preprocess image for feature extraction
            if isinstance(image, np.ndarray):
                # Assume image is already preprocessed
                processed_image = image
            else:
                # Preprocess PIL image
                image = image.resize((224, 224))
                processed_image = np.array(image) / 255.0
                processed_image = np.expand_dims(processed_image, axis=0)
            
            # Extract features
            features = self.feature_extractor.predict(processed_image, verbose=0)
            
            # Normalize features
            features = normalize(features, norm='l2')
            
            return features.flatten()
            
        except Exception as e:
            logger.error(f"Feature extraction error: {str(e)}")
            raise
    
    def add_item(self, item_id, image, metadata=None):
        """Add new item to similarity database"""
        try:
            # Extract features
            features = self.extract_features(image)
            
            # Store features and metadata
            self.feature_database[item_id] = features
            self.metadata_database[item_id] = metadata or {}
            
            # Rebuild similarity matrix
            self.build_similarity_matrix()
            
            # Save database
            self.save_database()
            
            logger.info(f"✅ Added item {item_id} to similarity database")
            
        except Exception as e:
            logger.error(f"Error adding item to similarity database: {str(e)}")
            raise
    
    def build_similarity_matrix(self):
        """Build similarity matrix for all items"""
        try:
            if not self.feature_database:
                return
            
            # Get all features
            item_ids = list(self.feature_database.keys())
            features_matrix = np.vstack([self.feature_database[item_id] for item_id in item_ids])
            
            # Calculate similarity matrix
            self.similarity_matrix = cosine_similarity(features_matrix)
            
            logger.info(f"✅ Similarity matrix built: {self.similarity_matrix.shape}")
            
        except Exception as e:
            logger.error(f"Error building similarity matrix: {str(e)}")
    
    def find_similar_by_id(self, item_id, top_k=5, category_filter=None):
        """Find similar items by item ID"""
        try:
            if item_id not in self.feature_database:
                raise Exception(f"Item {item_id} not found in database")
            
            if self.similarity_matrix is None:
                raise Exception("Similarity matrix not built")
            
            # Get item index
            item_ids = list(self.feature_database.keys())
            item_index = item_ids.index(item_id)
            
            # Get similarity scores
            similarities = self.similarity_matrix[item_index]
            
            # Get indices of most similar items (excluding self)
            similar_indices = np.argsort(similarities)[::-1][1:top_k+1]
            
            # Build results
            results = []
            for idx in similar_indices:
                similar_item_id = item_ids[idx]
                metadata = self.metadata_database.get(similar_item_id, {})
                
                # Apply category filter if specified
                if category_filter and metadata.get('category') != category_filter:
                    continue
                
                results.append({
                    'item_id': similar_item_id,
                    'similarity_score': float(similarities[idx]),
                    'metadata': metadata
                })
            
            return {
                'items': results[:top_k],
                'total': len(results),
                'scores': [r['similarity_score'] for r in results[:top_k]]
            }
            
        except Exception as e:
            logger.error(f"Error finding similar items: {str(e)}")
            raise
    
    def find_similar_by_image(self, image, top_k=5, category_filter=None):
        """Find similar items by uploaded image"""
        try:
            # Extract features from query image
            query_features = self.extract_features(image)
            
            # Calculate similarities with all items
            similarities = {}
            for item_id, features in self.feature_database.items():
                similarity = cosine_similarity(
                    query_features.reshape(1, -1), 
                    features.reshape(1, -1)
                )[0][0]
                similarities[item_id] = similarity
            
            # Sort by similarity
            sorted_items = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
            
            # Build results
            results = []
            for item_id, similarity_score in sorted_items[:top_k]:
                metadata = self.metadata_database.get(item_id, {})
                
                # Apply category filter if specified
                if category_filter and metadata.get('category') != category_filter:
                    continue
                
                results.append({
                    'item_id': item_id,
                    'similarity_score': float(similarity_score),
                    'metadata': metadata
                })
            
            return {
                'items': results[:top_k],
                'total': len(results),
                'scores': [r['similarity_score'] for r in results[:top_k]]
            }
            
        except Exception as e:
            logger.error(f"Error finding similar items by image: {str(e)}")
            raise
    
    def get_item_features(self, item_id):
        """Get features for a specific item"""
        if item_id in self.feature_database:
            return self.feature_database[item_id]
        return None
    
    def save_database(self):
        """Save similarity database to file"""
        try:
            # Convert numpy arrays to lists for JSON serialization
            features_serializable = {}
            for item_id, features in self.feature_database.items():
                features_serializable[item_id] = features.tolist()
            
            data = {
                'features': features_serializable,
                'metadata': self.metadata_database,
                'last_updated': datetime.now().isoformat(),
                'total_items': len(self.feature_database)
            }
            
            with open(self.database_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"✅ Similarity database saved: {len(self.feature_database)} items")
            
        except Exception as e:
            logger.error(f"Error saving similarity database: {str(e)}")
    
    def get_statistics(self):
        """Get similarity engine statistics"""
        return {
            'total_items': len(self.feature_database),
            'categories': list(set(
                metadata.get('category', 'unknown') 
                for metadata in self.metadata_database.values()
            )),
            'feature_dimension': (
                len(next(iter(self.feature_database.values()))) 
                if self.feature_database else 0
            ),
            'similarity_matrix_shape': (
                self.similarity_matrix.shape if self.similarity_matrix is not None else None
            ),
            'database_path': self.database_path
        }
