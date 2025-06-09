# services/attributes_service.py - Multi-label Attribute Analysis Service
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import os
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class AttributeAnalyzer:
    """Multi-label Fashion Attribute Analysis Service"""
    
    def __init__(self):
        self.models = {}
        self.attribute_categories = {
            'colors': ['black', 'white', 'red', 'blue', 'green', 'yellow', 'pink', 'purple', 'brown', 'gray', 'orange', 'navy', 'beige'],
            'patterns': ['solid', 'striped', 'plaid', 'floral', 'polka_dot', 'animal_print', 'geometric', 'abstract'],
            'materials': ['cotton', 'polyester', 'wool', 'silk', 'denim', 'leather', 'linen', 'cashmere', 'synthetic'],
            'styles': ['casual', 'formal', 'business', 'party', 'sport', 'vintage', 'bohemian', 'minimalist'],
            'fits': ['tight', 'fitted', 'regular', 'loose', 'oversized'],
            'lengths': ['mini', 'short', 'knee_length', 'midi', 'maxi', 'ankle_length'],
            'sleeves': ['sleeveless', 'short_sleeve', 'three_quarter', 'long_sleeve'],
            'necklines': ['crew_neck', 'v_neck', 'scoop_neck', 'high_neck', 'off_shoulder', 'strapless']
        }
        
        # Initialize models
        self.setup_attribute_models()
    
    def setup_attribute_models(self):
        """Setup multi-label attribute classification models"""
        try:
            # For this implementation, we'll use a single model that outputs multiple attributes
            # In production, you might have separate models for each attribute category
            
            # Create a base feature extractor
            base_model = tf.keras.applications.EfficientNetV2B0(
                weights='imagenet',
                include_top=False,
                input_shape=(224, 224, 3)
            )
            
            # Freeze base model layers
            base_model.trainable = False
            
            # Add custom heads for each attribute category
            inputs = tf.keras.Input(shape=(224, 224, 3))
            x = base_model(inputs, training=False)
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            x = tf.keras.layers.Dropout(0.2)(x)
            
            # Create outputs for each attribute category
            outputs = {}
            for category, attributes in self.attribute_categories.items():
                # Multi-label classification head
                category_output = tf.keras.layers.Dense(
                    len(attributes), 
                    activation='sigmoid', 
                    name=f'{category}_output'
                )(x)
                outputs[category] = category_output
            
            # Create model
            self.attribute_model = tf.keras.Model(inputs, outputs)
            
            # In a real implementation, you would load pre-trained weights here
            # For demo purposes, we'll use the initialized model
            
            logger.info("✅ Attribute analysis models initialized")
            
        except Exception as e:
            logger.error(f"❌ Error setting up attribute models: {str(e)}")
            raise
    
    def preprocess_image(self, image):
        """Preprocess image for attribute analysis"""
        try:
            # Ensure image is PIL Image
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize to model input size
            image = image.resize((224, 224), Image.Resampling.LANCZOS)
            
            # Convert to numpy array and normalize
            img_array = np.array(image, dtype=np.float32) / 255.0
            
            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            
            return img_array
            
        except Exception as e:
            logger.error(f"Error preprocessing image for attributes: {str(e)}")
            raise
    
    def analyze(self, image):
        """Analyze multiple attributes of a fashion item"""
        try:
            start_time = datetime.now()
            
            # Preprocess image
            processed_image = self.preprocess_image(image)
            
            # Get predictions for all attribute categories
            predictions = self.attribute_model.predict(processed_image, verbose=0)
            
            # Process predictions for each category
            analyzed_attributes = {}
            confidence_scores = {}
            
            for category, attributes in self.attribute_categories.items():
                category_predictions = predictions[category][0]
                
                # Get top attributes for this category
                top_indices = np.argsort(category_predictions)[::-1][:3]
                
                category_results = []
                category_confidences = []
                
                for idx in top_indices:
                    confidence = float(category_predictions[idx])
                    if confidence > 0.3:  # Threshold for inclusion
                        category_results.append({
                            'attribute': attributes[idx],
                            'confidence': confidence,
                            'percentage': confidence * 100
                        })
                        category_confidences.append(confidence)
                
                analyzed_attributes[category] = category_results
                confidence_scores[category] = np.mean(category_confidences) if category_confidences else 0.0
            
            # Enhanced analysis with computer vision techniques
            enhanced_analysis = self.enhanced_visual_analysis(image)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return {
                'attributes': analyzed_attributes,
                'confidence_scores': confidence_scores,
                'enhanced_analysis': enhanced_analysis,
                'overall_confidence': np.mean(list(confidence_scores.values())),
                'processing_time': processing_time,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Attribute analysis error: {str(e)}")
            raise
    
    def enhanced_visual_analysis(self, image):
        """Enhanced visual analysis using computer vision techniques"""
        try:
            # Convert PIL to OpenCV format
            if isinstance(image, Image.Image):
                cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            else:
                cv_image = image
            
            analysis = {}
            
            # Color analysis
            analysis['dominant_colors'] = self.analyze_dominant_colors(cv_image)
            
            # Texture analysis
            analysis['texture_features'] = self.analyze_texture(cv_image)
            
            # Pattern detection
            analysis['patterns'] = self.detect_patterns(cv_image)
            
            # Edge and contour analysis
            analysis['structural_features'] = self.analyze_structure(cv_image)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Enhanced visual analysis error: {str(e)}")
            return {}
    
    def analyze_dominant_colors(self, image):
        """Analyze dominant colors in the image"""
        try:
            # Reshape image to be a list of pixels
            data = image.reshape((-1, 3))
            data = np.float32(data)
            
            # Use K-means clustering to find dominant colors
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
            k = 5  # Number of dominant colors to find
            
            _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            # Convert centers to integers
            centers = np.uint8(centers)
            
            # Count frequency of each cluster
            unique, counts = np.unique(labels, return_counts=True)
            
            # Calculate percentages and create results
            total_pixels = len(labels)
            dominant_colors = []
            
            for i, center in enumerate(centers):
                percentage = (counts[i] / total_pixels) * 100
                # Convert BGR to RGB
                rgb_color = [int(center[2]), int(center[1]), int(center[0])]
                
                dominant_colors.append({
                    'rgb': rgb_color,
                    'hex': '#{:02x}{:02x}{:02x}'.format(*rgb_color),
                    'percentage': float(percentage)
                })
            
            # Sort by percentage
            dominant_colors.sort(key=lambda x: x['percentage'], reverse=True)
            
            return dominant_colors[:3]  # Return top 3 colors
            
        except Exception as e:
            logger.error(f"Color analysis error: {str(e)}")
            return []
    
    def analyze_texture(self, image):
        """Analyze texture features using Local Binary Patterns"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Calculate Local Binary Pattern
            radius = 3
            n_points = 8 * radius
            
            # Simplified LBP calculation
            lbp = np.zeros_like(gray)
            
            for i in range(radius, gray.shape[0] - radius):
                for j in range(radius, gray.shape[1] - radius):
                    center = gray[i, j]
                    code = 0
                    
                    # Sample points around the center
                    for k in range(n_points):
                        angle = 2 * np.pi * k / n_points
                        x = int(i + radius * np.cos(angle))
                        y = int(j + radius * np.sin(angle))
                        
                        if 0 <= x < gray.shape[0] and 0 <= y < gray.shape[1]:
                            if gray[x, y] >= center:
                                code |= (1 << k)
                    
                    lbp[i, j] = code
            
            # Calculate texture uniformity
            hist, _ = np.histogram(lbp.ravel(), bins=256, range=[0, 256])
            uniformity = np.sum(hist ** 2) / (lbp.size ** 2)
            
            # Calculate entropy (measure of randomness)
            hist_norm = hist / hist.sum()
            entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-10))
            
            return {
                'uniformity': float(uniformity),
                'entropy': float(entropy),
                'texture_category': self.classify_texture(uniformity, entropy)
            }
            
        except Exception as e:
            logger.error(f"Texture analysis error: {str(e)}")
            return {}
    
    def classify_texture(self, uniformity, entropy):
        """Classify texture based on uniformity and entropy"""
        if uniformity > 0.1:
            return 'smooth'
        elif entropy > 6:
            return 'rough'
        elif 4 < entropy <= 6:
            return 'medium'
        else:
            return 'fine'
    
    def detect_patterns(self, image):
        """Detect patterns like stripes, checks, etc."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Line detection using Hough Transform
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            pattern_info = {
                'has_lines': lines is not None and len(lines) > 10,
                'line_count': len(lines) if lines is not None else 0
            }
            
            # Analyze line orientations
            if lines is not None and len(lines) > 0:
                angles = []
                for line in lines:
                    rho, theta = line[0]
                    angle = theta * 180 / np.pi
                    angles.append(angle)
                
                angles = np.array(angles)
                
                # Check for horizontal/vertical dominance (stripes)
                horizontal_lines = np.sum((angles < 10) | (angles > 170))
                vertical_lines = np.sum((85 < angles) & (angles < 95))
                
                pattern_info['pattern_type'] = 'stripes' if (horizontal_lines > 5 or vertical_lines > 5) else 'complex'
                pattern_info['orientation'] = 'horizontal' if horizontal_lines > vertical_lines else 'vertical'
            else:
                pattern_info['pattern_type'] = 'solid'
                pattern_info['orientation'] = 'none'
            
            return pattern_info
            
        except Exception as e:
            logger.error(f"Pattern detection error: {str(e)}")
            return {}
    
    def analyze_structure(self, image):
        """Analyze structural features like edges and contours"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Calculate structural features
            edge_density = np.sum(edges > 0) / edges.size
            contour_count = len(contours)
            
            # Calculate largest contour area
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                largest_area = cv2.contourArea(largest_contour)
                relative_area = largest_area / (image.shape[0] * image.shape[1])
            else:
                relative_area = 0
            
            return {
                'edge_density': float(edge_density),
                'contour_count': contour_count,
                'largest_contour_area_ratio': float(relative_area),
                'structural_complexity': 'high' if edge_density > 0.1 else 'medium' if edge_density > 0.05 else 'low'
            }
            
        except Exception as e:
            logger.error(f"Structure analysis error: {str(e)}")
            return {}
    
    def get_attribute_categories(self):
        """Get all available attribute categories"""
        return self.attribute_categories
    
    def get_statistics(self):
        """Get attribute analyzer statistics"""
        return {
            'attribute_categories': list(self.attribute_categories.keys()),
            'total_attributes': sum(len(attrs) for attrs in self.attribute_categories.values()),
            'model_loaded': hasattr(self, 'attribute_model'),
            'supported_analyses': ['colors', 'patterns', 'materials', 'styles', 'texture', 'structure']
        }
