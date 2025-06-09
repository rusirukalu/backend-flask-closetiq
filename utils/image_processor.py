# utils/image_processor.py - Enhanced Image Processing Utilities
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import io
import logging
from typing import Tuple, Optional, Union
import base64

logger = logging.getLogger(__name__)

class ImageProcessor:
    """Advanced Image Processing Utilities for Fashion AI"""
    
    def __init__(self):
        self.target_size = (224, 224)
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    
    def load_and_preprocess(self, image_input, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Load and preprocess image for AI model input"""
        try:
            # Handle different input types
            if hasattr(image_input, 'read'):
                # File-like object
                image_bytes = image_input.read()
                image = Image.open(io.BytesIO(image_bytes))
            elif isinstance(image_input, str):
                if image_input.startswith('data:image'):
                    # Base64 encoded image
                    image = self.decode_base64_image(image_input)
                else:
                    # File path
                    image = Image.open(image_input)
            elif isinstance(image_input, np.ndarray):
                # NumPy array
                image = Image.fromarray(image_input)
            elif isinstance(image_input, Image.Image):
                # PIL Image
                image = image_input
            else:
                raise ValueError(f"Unsupported image input type: {type(image_input)}")
            
            # Validate image size
            if image.size[0] < 10 or image.size[1] < 10:
                raise ValueError(f"Image too small: {image.size}")
            
            # Preprocess image
            processed_image = self.preprocess_image(image, target_size or self.target_size)
            
            return processed_image
            
        except Exception as e:
            logger.error(f"Image loading and preprocessing error: {str(e)}")
            raise
    
    def preprocess_image(self, image: Image.Image, target_size: Tuple[int, int]) -> np.ndarray:
        """Preprocess PIL image for model input"""
        try:
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Apply image enhancements
            enhanced_image = self.enhance_image(image)
            
            # Resize image
            resized_image = enhanced_image.resize(target_size, Image.Resampling.LANCZOS)
            
            # Convert to numpy array and normalize
            img_array = np.array(resized_image, dtype=np.float32)
            img_array = img_array / 255.0
            
            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            
            return img_array
            
        except Exception as e:
            logger.error(f"Image preprocessing error: {str(e)}")
            raise
    
    def enhance_image(self, image: Image.Image) -> Image.Image:
        """Apply image enhancements for better AI processing"""
        try:
            # Auto-adjust brightness and contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.1)  # Slight contrast boost
            
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(1.05)  # Slight brightness boost
            
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.1)  # Slight sharpness boost
            
            return image
            
        except Exception as e:
            logger.error(f"Image enhancement error: {str(e)}")
            return image  # Return original if enhancement fails
    
    def decode_base64_image(self, base64_string: str) -> Image.Image:
        """Decode base64 encoded image"""
        try:
            # Remove data URL prefix if present
            if base64_string.startswith('data:image'):
                base64_string = base64_string.split(',')[1]
            
            # Decode base64
            image_data = base64.b64decode(base64_string)
            
            # Create PIL image
            image = Image.open(io.BytesIO(image_data))
            
            return image
            
        except Exception as e:
            logger.error(f"Base64 image decoding error: {str(e)}")
            raise
    
    def extract_image_features(self, image: Union[Image.Image, np.ndarray]) -> dict:
        """Extract various features from image for analysis"""
        try:
            # Convert to OpenCV format
            if isinstance(image, Image.Image):
                cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            else:
                cv_image = image
            
            features = {}
            
            # Basic image properties
            height, width = cv_image.shape[:2]
            features['dimensions'] = {'width': width, 'height': height}
            features['aspect_ratio'] = width / height
            
            # Color analysis
            features['color_analysis'] = self.analyze_colors(cv_image)
            
            # Texture analysis
            features['texture_analysis'] = self.analyze_texture(cv_image)
            
            # Edge analysis
            features['edge_analysis'] = self.analyze_edges(cv_image)
            
            # Brightness and contrast
            features['brightness_contrast'] = self.analyze_brightness_contrast(cv_image)
            
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction error: {str(e)}")
            return {}
    
    def analyze_colors(self, cv_image: np.ndarray) -> dict:
        """Analyze color properties of the image"""
        try:
            # Convert to RGB for analysis
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            
            # Calculate color statistics
            mean_color = np.mean(rgb_image, axis=(0, 1))
            std_color = np.std(rgb_image, axis=(0, 1))
            
            # Dominant colors using K-means
            pixels = rgb_image.reshape(-1, 3)
            from sklearn.cluster import KMeans
            
            # Use 5 clusters for dominant colors
            kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
            kmeans.fit(pixels)
            
            dominant_colors = []
            labels = kmeans.labels_
            
            for i, color in enumerate(kmeans.cluster_centers_):
                count = np.sum(labels == i)
                percentage = (count / len(labels)) * 100
                
                dominant_colors.append({
                    'color': color.astype(int).tolist(),
                    'percentage': float(percentage)
                })
            
            # Sort by percentage
            dominant_colors.sort(key=lambda x: x['percentage'], reverse=True)
            
            return {
                'mean_color': mean_color.tolist(),
                'color_std': std_color.tolist(),
                'dominant_colors': dominant_colors[:3]  # Top 3 colors
            }
            
        except Exception as e:
            logger.error(f"Color analysis error: {str(e)}")
            return {}
    
    def analyze_texture(self, cv_image: np.ndarray) -> dict:
        """Analyze texture properties"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Calculate texture features using Local Binary Patterns
            # Simplified implementation
            texture_variance = float(np.var(gray))
            texture_entropy = self.calculate_entropy(gray)
            
            # Edge density as texture measure
            edges = cv2.Canny(gray, 50, 150)
            edge_density = float(np.sum(edges > 0) / edges.size)
            
            return {
                'variance': texture_variance,
                'entropy': texture_entropy,
                'edge_density': edge_density,
                'texture_class': self.classify_texture(texture_variance, edge_density)
            }
            
        except Exception as e:
            logger.error(f"Texture analysis error: {str(e)}")
            return {}
    
    def calculate_entropy(self, gray_image: np.ndarray) -> float:
        """Calculate image entropy"""
        try:
            # Calculate histogram
            hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
            hist = hist.flatten()
            
            # Normalize
            hist = hist / hist.sum()
            
            # Calculate entropy
            entropy = -np.sum(hist * np.log2(hist + 1e-10))
            
            return float(entropy)
            
        except Exception as e:
            logger.error(f"Entropy calculation error: {str(e)}")
            return 0.0
    
    def classify_texture(self, variance: float, edge_density: float) -> str:
        """Classify texture based on features"""
        if variance > 2000 and edge_density > 0.1:
            return 'rough'
        elif variance > 1000:
            return 'medium'
        else:
            return 'smooth'
    
    def analyze_edges(self, cv_image: np.ndarray) -> dict:
        """Analyze edge properties"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Line detection
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            # Calculate edge statistics
            edge_count = np.sum(edges > 0)
            edge_density = edge_count / edges.size
            
            line_count = len(lines) if lines is not None else 0
            
            return {
                'edge_count': int(edge_count),
                'edge_density': float(edge_density),
                'line_count': line_count,
                'has_strong_lines': line_count > 10
            }
            
        except Exception as e:
            logger.error(f"Edge analysis error: {str(e)}")
            return {}
    
    def analyze_brightness_contrast(self, cv_image: np.ndarray) -> dict:
        """Analyze brightness and contrast"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Calculate brightness (mean intensity)
            brightness = float(np.mean(gray))
            
            # Calculate contrast (standard deviation)
            contrast = float(np.std(gray))
            
            # Classify brightness
            if brightness < 85:
                brightness_class = 'dark'
            elif brightness > 170:
                brightness_class = 'bright'
            else:
                brightness_class = 'normal'
            
            # Classify contrast
            if contrast < 30:
                contrast_class = 'low'
            elif contrast > 80:
                contrast_class = 'high'
            else:
                contrast_class = 'normal'
            
            return {
                'brightness': brightness,
                'contrast': contrast,
                'brightness_class': brightness_class,
                'contrast_class': contrast_class
            }
            
        except Exception as e:
            logger.error(f"Brightness/contrast analysis error: {str(e)}")
            return {}
    
    def detect_fashion_regions(self, image: Union[Image.Image, np.ndarray]) -> dict:
        """Detect fashion item regions in the image"""
        try:
            # Convert to OpenCV format
            if isinstance(image, Image.Image):
                cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            else:
                cv_image = image
            
            # Simple region detection using contours
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Find largest contour (assume it's the main fashion item)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Get bounding box
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Calculate area ratio
                total_area = cv_image.shape[0] * cv_image.shape[1]
                item_area = w * h
                area_ratio = item_area / total_area
                
                return {
                    'bounding_box': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)},
                    'area_ratio': float(area_ratio),
                    'contour_count': len(contours),
                    'item_detected': area_ratio > 0.1  # Item takes up at least 10% of image
                }
            else:
                return {
                    'bounding_box': None,
                    'area_ratio': 0.0,
                    'contour_count': 0,
                    'item_detected': False
                }
                
        except Exception as e:
            logger.error(f"Fashion region detection error: {str(e)}")
            return {}
    
    def generate_image_quality_score(self, image: Union[Image.Image, np.ndarray]) -> dict:
        """Generate overall image quality score for AI processing"""
        try:
            # Convert input to appropriate format
            if isinstance(image, np.ndarray):
                # Validate array shape
                if image.ndim == 4:  # Shape (1, height, width, 3)
                    image = image.squeeze(0)  # Remove batch dimension
                elif image.ndim != 3:  # Must be (height, width, 3)
                    raise ValueError(f"Unsupported array shape: {image.shape}")
                
                # Validate dimensions
                height, width = image.shape[:2]
                if height < 10 or width < 10:
                    raise ValueError(f"Image array too small: {width}x{height}")
                
                # Convert float32 array back to uint8 for OpenCV/PIL
                if image.dtype == np.float32:
                    image = (image * 255).astype(np.uint8)
                image = Image.fromarray(image)  # Convert to Pillow Image
            
            # Validate Pillow Image size
            width, height = image.size
            if width < 10 or height < 10:
                raise ValueError(f"Image too small: {width}x{height}")
            
            logger.info(f"Processing image for quality score with size: {width}x{height}")
            
            features = self.extract_image_features(image)
            region_info = self.detect_fashion_regions(image)
            
            # Calculate quality components
            scores = {}
            
            # Resolution score
            dimensions = features.get('dimensions', {})
            width = dimensions.get('width', 0)
            height = dimensions.get('height', 0)
            resolution_score = min(1.0, (width * height) / (512 * 512))  # Normalize to 512x512
            scores['resolution'] = resolution_score
            
            # Brightness score
            brightness_contrast = features.get('brightness_contrast', {})
            brightness = brightness_contrast.get('brightness', 128)
            brightness_score = 1.0 - abs(brightness - 128) / 128  # Optimal around 128
            scores['brightness'] = max(0.0, brightness_score)
            
            # Contrast score
            contrast = brightness_contrast.get('contrast', 50)
            contrast_score = min(1.0, contrast / 80)  # Good contrast around 80
            scores['contrast'] = contrast_score
            
            # Item detection score
            item_score = 1.0 if region_info.get('item_detected', False) else 0.3
            scores['item_detection'] = item_score
            
            # Overall quality score (weighted average)
            weights = {
                'resolution': 0.2,
                'brightness': 0.25,
                'contrast': 0.25,
                'item_detection': 0.3
            }
            
            overall_score = sum(scores[component] * weights[component] for component in weights)
            
            # Quality grade
            if overall_score >= 0.9:
                grade = 'Excellent'
            elif overall_score >= 0.8:
                grade = 'Very Good'
            elif overall_score >= 0.7:
                grade = 'Good'
            elif overall_score >= 0.6:
                grade = 'Fair'
            else:
                grade = 'Poor'
            
            return {
                'overall_score': float(overall_score),
                'grade': grade,
                'component_scores': scores,
                'recommendations': self._generate_quality_recommendations(scores)
            }
            
        except Exception as e:
            logger.error(f"Quality score generation error: {str(e)}")
            raise
    
    def _generate_quality_recommendations(self, scores: dict) -> list:
        """Generate recommendations for improving image quality"""
        recommendations = []
        
        if scores.get('resolution', 1.0) < 0.7:
            recommendations.append("Use a higher resolution image for better results")
        
        if scores.get('brightness', 1.0) < 0.6:
            recommendations.append("Improve lighting conditions - image may be too dark or bright")
        
        if scores.get('contrast', 1.0) < 0.6:
            recommendations.append("Increase contrast for better feature detection")
        
        if scores.get('item_detection', 1.0) < 0.8:
            recommendations.append("Ensure the fashion item fills most of the frame")
        
        if not recommendations:
            recommendations.append("Image quality is good for AI processing")
        
        return recommendations
    
    def batch_process_images(self, image_inputs: list, target_size: Optional[Tuple[int, int]] = None) -> list:
        """Process multiple images in batch"""
        try:
            processed_images = []
            
            for i, image_input in enumerate(image_inputs):
                try:
                    processed_image = self.load_and_preprocess(image_input, target_size)
                    processed_images.append({
                        'index': i,
                        'success': True,
                        'image': processed_image
                    })
                except Exception as e:
                    logger.error(f"Error processing image {i}: {str(e)}")
                    processed_images.append({
                        'index': i,
                        'success': False,
                        'error': str(e)
                    })
            
            return processed_images
            
        except Exception as e:
            logger.error(f"Batch processing error: {str(e)}")
            raise