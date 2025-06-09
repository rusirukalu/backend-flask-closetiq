# services/classification_service.py - Enhanced Fashion Classification Service
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import os
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class FashionClassifier:
    """Enhanced Fashion Classification Service with CAM analysis"""
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.model_version = "1.0.0"
        self.categories = [
            'shirts_blouses', 'tshirts_tops', 'dresses', 'pants_jeans',
            'shorts', 'skirts', 'jackets_coats', 'sweaters', 
            'shoes_sneakers', 'shoes_formal', 'bags_accessories'
        ]
        
        # Load model on initialization
        self.load_model()
        
        # Performance tracking
        self.classification_count = 0
        self.total_processing_time = 0
    
    def load_model(self):
        """Load the enhanced fashion classification model"""
        try:
            if os.path.exists(self.model_path):
                self.model = tf.keras.models.load_model(self.model_path)
                logger.info(f"✅ Enhanced model loaded from {self.model_path}")
                logger.info(f"Model input shape: {self.model.input_shape}")
                logger.info(f"Model output shape: {self.model.output_shape}")
                logger.info("Model architecture:")
                self.model.summary(print_fn=lambda x: logger.info(x))
                return True
            else:
                logger.error(f"❌ Model file not found: {self.model_path}")
                return False
        except Exception as e:
            logger.error(f"❌ Error loading model: {str(e)}")
            return False
    
    def is_loaded(self):
        """Check if model is loaded"""
        return self.model is not None
    
    def preprocess_image(self, image):
        """Preprocess image for classification"""
        try:
            # Validate input
            if not isinstance(image, Image.Image):
                if isinstance(image, np.ndarray):
                    image = Image.fromarray(image.astype(np.uint8))
                else:
                    raise ValueError("Input must be a PIL Image or NumPy array")
            
            # Validate image
            if image.size[0] == 0 or image.size[1] == 0:
                raise ValueError("Invalid image: zero width or height")
            
            # Log original image details
            logger.info(f"Original image mode: {image.mode}, size: {image.size}")
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                if image.mode == 'L':  # Grayscale
                    image = image.convert('RGB')
                elif image.mode == 'RGBA':  # RGBA
                    image = image.convert('RGB')
                else:
                    logger.warning(f"Unexpected image mode: {image.mode}, converting to RGB")
                    image = image.convert('RGB')
            
            # Resize to model input size
            image = image.resize((224, 224), Image.Resampling.LANCZOS)
            
            # Convert to numpy array and normalize
            img_array = np.array(image, dtype=np.float32) / 255.0
            
            # Ensure correct shape (224, 224, 3)
            if img_array.shape != (224, 224, 3):
                logger.error(f"Unexpected image shape after preprocessing: {img_array.shape}")
                raise ValueError(f"Unexpected image shape: {img_array.shape}")
            
            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            
            # Validate final shape
            if img_array.shape != (1, 224, 224, 3):
                logger.error(f"Final image shape incorrect: {img_array.shape}")
                raise ValueError(f"Final image shape incorrect: {img_array.shape}")
            
            # Log final shape and dtype
            logger.info(f"Preprocessed image shape: {img_array.shape}, dtype: {img_array.dtype}")
            
            return img_array
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise
    
    def classify(self, image):
        """Classify fashion image with enhanced features"""
        try:
            if not self.is_loaded():
                raise Exception("Model not loaded")
            
            start_time = datetime.now()
            
            # Preprocess image
            processed_image = self.preprocess_image(image)
            
            # Get predictions
            predictions = self.model.predict(processed_image, verbose=0)
            
            # Get top prediction
            predicted_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_idx])
            predicted_category = self.categories[predicted_idx]
            
            # Get all predictions
            all_predictions = []
            for i, score in enumerate(predictions[0]):
                all_predictions.append({
                    'class': self.categories[i],
                    'confidence': float(score),
                    'percentage': float(score * 100)
                })
            
            # Sort by confidence
            all_predictions.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Calculate quality score
            quality_score = self.calculate_quality_score(predictions[0])
            
            # Generate Class Activation Maps (if supported by model)
            cam_analysis = self.generate_cam_analysis(processed_image, predicted_idx)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Update performance tracking
            self.classification_count += 1
            self.total_processing_time += processing_time
            
            return {
                'predicted_category': predicted_category,
                'predicted_class': predicted_category,
                'confidence': confidence,
                'all_predictions': all_predictions,
                'quality_score': quality_score,
                'processing_time': processing_time,
                'cam_analysis': cam_analysis,
                'model_version': self.model_version,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Classification error: {str(e)}")
            raise
    
    def calculate_quality_score(self, predictions):
        """Calculate image quality score based on prediction confidence"""
        max_confidence = np.max(predictions)
        second_max = np.partition(predictions, -2)[-2]
        
        # Quality based on confidence gap
        confidence_gap = max_confidence - second_max
        
        if confidence_gap > 0.5:
            return {'score': 'Excellent', 'value': 95}
        elif confidence_gap > 0.3:
            return {'score': 'Very Good', 'value': 85}
        elif confidence_gap > 0.2:
            return {'score': 'Good', 'value': 75}
        elif confidence_gap > 0.1:
            return {'score': 'Fair', 'value': 65}
        else:
            return {'score': 'Poor', 'value': 45}
    
    def generate_cam_analysis(self, processed_image, predicted_class):
        """Generate Class Activation Maps for better interpretability"""
        try:
            # Get the last convolutional layer
            last_conv_layer_name = None
            for layer in reversed(self.model.layers):
                if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.SeparableConv2D)):
                    last_conv_layer_name = layer.name
                    break
            
            if not last_conv_layer_name:
                return {"available": False, "reason": "No convolutional layers found"}
            
            # Create a model that maps the input image to the activations
            # of the last conv layer as well as the output predictions
            grad_model = tf.keras.models.Model(
                self.model.inputs,
                [self.model.get_layer(last_conv_layer_name).output, self.model.output]
            )
            
            # Compute gradients
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(processed_image)
                loss = predictions[:, predicted_class]
            
            # Get gradients
            grads = tape.gradient(loss, conv_outputs)
            
            # Global average pooling of gradients
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            
            # Weight the feature maps by the gradients
            conv_outputs = conv_outputs[0]
            heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
            
            # Normalize heatmap
            heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
            
            return {
                "available": True,
                "heatmap_shape": heatmap.shape.as_list(),
                "attention_score": float(tf.reduce_mean(heatmap)),
                "max_activation": float(tf.reduce_max(heatmap)),
                "layer_used": last_conv_layer_name
            }
            
        except Exception as e:
            logger.error(f"CAM analysis error: {str(e)}")
            return {"available": False, "reason": str(e)}
    
    def get_performance_metrics(self):
        """Get model performance metrics"""
        avg_processing_time = (
            self.total_processing_time / self.classification_count
            if self.classification_count > 0 else 0
        )
        
        return {
            'total_classifications': self.classification_count,
            'average_processing_time_ms': avg_processing_time,
            'model_version': self.model_version,
            'categories_count': len(self.categories),
            'model_loaded': self.is_loaded()
        }
    
    def batch_classify(self, images):
        """Classify multiple images in batch"""
        try:
            results = []
            
            for i, image in enumerate(images):
                try:
                    result = self.classify(image)
                    result['batch_index'] = i
                    results.append(result)
                except Exception as e:
                    results.append({
                        'batch_index': i,
                        'success': False,
                        'error': str(e)
                    })
            
            return {
                'results': results,
                'total_processed': len(results),
                'successful': len([r for r in results if r.get('success', True)])
            }
            
        except Exception as e:
            logger.error(f"Batch classification error: {str(e)}")
            raise