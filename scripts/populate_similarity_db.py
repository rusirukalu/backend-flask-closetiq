# app.py - Complete Fashion AI Backend Application with Training Integration
import os
import sys
import logging
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from datetime import datetime
from PIL import Image
import traceback
import uuid
import threading
import atexit
from pathlib import Path

# Import our services
from services.classification_service import FashionClassifier
from services.similarity_service import SimilarityEngine
from services.attributes_service import AttributeAnalyzer
from services.compatibility_service import StyleCompatibilityEngine
from services.knowledge_service import FashionKnowledgeService
from services.recommendation_service import OutfitRecommendationEngine
from services.training_service import TrainingOrchestrator
from utils.image_processor import ImageProcessor
from utils.model_manager import ModelManager
from config.settings import Config

# Import training API blueprint
from api.training import training_bp

# Configure logging
def setup_logging():
    """Setup comprehensive logging"""
    log_format = logging.Formatter(Config.LOG_FORMAT)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    
    # File handler
    file_handler = logging.FileHandler(Config.LOGS_DIR / 'fashion_ai_backend.log')
    file_handler.setFormatter(log_format)
    
    # Setup root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, Config.LOG_LEVEL))
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

# Initialize logging
logger = setup_logging()

class FashionAIBackend:
    """Main Fashion AI Backend Application with Training Capabilities"""
    
    def __init__(self):
        self.app = None
        self.services = {}
        self.startup_time = datetime.now()
        self.request_count = 0
        self.error_count = 0
        
    def create_app(self):
        """Create and configure the Flask application"""
        try:
            logger.info("🚀 Initializing Fashion AI Backend with Training...")
            
            # Validate configuration
            config_errors = Config.validate_config()
            if config_errors:
                for error in config_errors:
                    logger.error(f"❌ Configuration error: {error}")
                raise Exception("Configuration validation failed")
            
            # Create Flask app
            self.app = Flask(__name__)
            self.app.config.from_object(Config)
            
            # Enable CORS for all API routes, explicitly allowing localhost:8000
            CORS(self.app, resources={r"/api/*": {"origins": "http://localhost:8000"}})
            
            # Initialize services
            self._initialize_services()
            
            # Register blueprints
            self._register_blueprints()
            
            # Setup routes
            self._setup_routes()
            
            # Setup error handlers
            self._setup_error_handlers()
            
            # Setup request hooks
            self._setup_request_hooks()
            
            logger.info("✅ Fashion AI Backend with Training initialized successfully")
            return self.app
            
        except Exception as e:
            logger.error(f"❌ Failed to create application: {str(e)}")
            raise
    
    def _initialize_services(self):
        """Initialize all AI services including training"""
        try:
            logger.info("🔧 Initializing AI services...")
            
            # Initialize model manager first
            self.services['model_manager'] = ModelManager()
            
            # Initialize image processor
            self.services['image_processor'] = ImageProcessor()
            
            # Initialize classification service
            logger.info("   Loading classification service...")
            self.services['classifier'] = FashionClassifier(Config.BASE_MODEL_PATH)
            
            # Initialize training orchestrator
            logger.info("   Loading training orchestrator...")
            self.services['training_orchestrator'] = TrainingOrchestrator()
            
            # Initialize similarity engine (if enabled)
            if Config.ENABLE_SIMILARITY_SEARCH:
                logger.info("   Loading similarity engine...")
                self.services['similarity_engine'] = SimilarityEngine(Config.SIMILARITY_DB_PATH)
            
            # Initialize attribute analyzer (if enabled)
            if Config.ENABLE_ATTRIBUTE_ANALYSIS:
                logger.info("   Loading attribute analyzer...")
                self.services['attribute_analyzer'] = AttributeAnalyzer()
            
            # Initialize compatibility engine
            logger.info("   Loading compatibility engine...")
            self.services['compatibility_engine'] = StyleCompatibilityEngine()
            
            # Initialize knowledge graph (if enabled)
            if Config.ENABLE_KNOWLEDGE_GRAPH:
                logger.info("   Loading knowledge graph...")
                self.services['knowledge_service'] = FashionKnowledgeService(Config.KNOWLEDGE_GRAPH_PATH)
            
            # Initialize recommendation engine
            logger.info("   Loading recommendation engine...")
            self.services['recommendation_engine'] = OutfitRecommendationEngine()
            
            logger.info("✅ All AI services initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Service initialization failed: {str(e)}")
            raise
    
    def _register_blueprints(self):
        """Register API blueprints"""
        try:
            # Register training API blueprint
            self.app.register_blueprint(training_bp, url_prefix='/api/training')
            logger.info("📚 Training API blueprint registered")
            
        except Exception as e:
            logger.error(f"❌ Blueprint registration failed: {str(e)}")
            raise
    
    def _setup_routes(self):
        """Setup all API routes"""
        
        # Health and status endpoints
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """Comprehensive health check including training status"""
            try:
                service_status = {}
                for name, service in self.services.items():
                    if hasattr(service, 'is_loaded'):
                        service_status[name] = service.is_loaded()
                    else:
                        service_status[name] = True
                
                uptime = (datetime.now() - self.startup_time).total_seconds()
                
                return jsonify({
                    'success': True,
                    'message': 'Fashion AI Backend with Training is running',
                    'timestamp': datetime.now().isoformat(),
                    'uptime_seconds': uptime,
                    'version': Config.MODEL_VERSION,
                    'services': service_status,
                    'request_count': self.request_count,
                    'error_count': self.error_count,
                    'error_rate': (self.error_count / max(self.request_count, 1)) * 100,
                    'features': {
                        'training_enabled': True,
                        'retraining_enabled': True,
                        'model_versioning': True
                    }
                })
                
            except Exception as e:
                logger.error(f"Health check error: {str(e)}")
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/api/status', methods=['GET'])
        def detailed_status():
            """Detailed system status including training capabilities"""
            try:
                status = self.services['model_manager'].get_status()
                
                # Add training-specific status
                training_status = {
                    'available_models': len(list(Path('models/trained').glob('*.keras'))),
                    'training_scripts': len(list(Path('scripts/training').glob('*.py'))),
                    'datasets_available': len(list(Path('datasets').iterdir())),
                    'last_training': self._get_last_training_info()
                }
                
                status['training_system'] = training_status
                
                return jsonify({
                    'success': True,
                    'detailed_status': status,
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500
        
        # Classification endpoints
        @self.app.route('/api/classify', methods=['POST'])
        def classify_image():
            """Enhanced image classification with comprehensive analysis"""
            try:
                # Validate request
                if 'file' not in request.files:
                    return jsonify({'success': False, 'error': 'No file provided'}), 400
                
                file = request.files['file']
                if file.filename == '':
                    return jsonify({'success': False, 'error': 'No file selected'}), 400
                
                # Validate file format
                if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    return jsonify({'success': False, 'error': 'Unsupported file format. Use JPG or PNG'}), 400
                
                # Load raw image
                file.seek(0)
                raw_image = Image.open(file).convert('RGB')
                logger.info(f"Raw image size: {raw_image.size}, mode: {raw_image.mode}, type: {type(raw_image)}")
                
                # Process image for classification
                start_time = datetime.now()
                # Pass raw PIL Image to classifier directly
                classification_result = self.services['classifier'].classify(raw_image)
                
                # Get image quality score using raw image
                file.seek(0)
                raw_image = Image.open(file).convert('RGB')
                quality_info = self.services['image_processor'].generate_image_quality_score(raw_image)
                
                # Get additional attributes if enabled
                attributes = {}
                if Config.ENABLE_ATTRIBUTE_ANALYSIS and 'attribute_analyzer' in self.services:
                    # Use raw image for attribute analysis
                    attributes = self.services['attribute_analyzer'].analyze(raw_image)
                
                processing_time = (datetime.now() - start_time).total_seconds() * 1000
                
                return jsonify({
                    'success': True,
                    'classification': classification_result,
                    'image_quality': quality_info,
                    'attributes': attributes,
                    'processing_time_ms': processing_time,
                    'model_version': Config.MODEL_VERSION,
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Classification error: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                return jsonify({'success': False, 'error': str(e)}), 500
        
        # Similarity search endpoint
        @self.app.route('/api/similarity/search', methods=['POST', 'OPTIONS'])
        def similarity_search():
            """Find similar items by item ID, category, or image"""
            if request.method == 'OPTIONS':
                return jsonify({'success': True}), 200

            try:
                logger.info(f"📝 POST /api/similarity/search - {request.remote_addr}")

                if not Config.ENABLE_SIMILARITY_SEARCH or 'similarity_engine' not in self.services:
                    return jsonify({'success': False, 'error': 'Similarity search is disabled'}), 503

                # Check if similarity database is empty
                if not self.services['similarity_engine'].is_loaded():
                    return jsonify({
                        'success': False,
                        'error': 'Similarity database is empty or not loaded',
                        'database_size': 0
                    }), 404

                # Handle JSON payload (itemId/category) or file upload
                item_id = None
                category = None
                image = None

                if request.is_json:
                    data = request.get_json()
                    item_id = data.get('itemId')
                    category = data.get('category', item_id)
                    limit = data.get('limit', 5)
                elif 'file' in request.files:
                    file = request.files['file']
                    if file.filename == '':
                        return jsonify({'success': False, 'error': 'No file selected'}), 400
                    if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                        return jsonify({'success': False, 'error': 'Unsupported file format'}), 400
                    file.seek(0)
                    image = Image.open(file).convert('RGB')
                    limit = int(request.form.get('limit', 5))
                    category = request.form.get('category')
                else:
                    return jsonify({'success': False, 'error': 'Invalid request: provide JSON or file'}), 400

                # Find similar items
                if image:
                    logger.info("Processing similarity search by image")
                    result = self.services['similarity_engine'].find_similar_by_image(
                        image=image,
                        top_k=limit,
                        category_filter=category
                    )
                else:
                    if not item_id and not category:
                        return jsonify({'success': False, 'error': 'Missing itemId or category'}), 400
                    item_id = item_id or category
                    logger.info(f"Processing similarity search by item_id: {item_id}")
                    result = self.services['similarity_engine'].find_similar_by_id(
                        item_id=item_id,
                        top_k=limit,
                        category_filter=category
                    )

                response = {
                    'success': True,
                    'results': result['items'],
                    'total': result['total'],
                    'scores': result['scores'],
                    'database_size': len(self.services['similarity_engine'].feature_database),
                    'timestamp': datetime.now().isoformat()
                }

                logger.info(f"📤 POST /api/similarity/search - 200 - {result['total']} items found")
                return jsonify(response), 200

            except Exception as e:
                logger.error(f"Similarity search error: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                return jsonify({'success': False, 'error': str(e), 'database_size': len(self.services['similarity_engine'].feature_database)}), 500

        # Model management with training integration
        @self.app.route('/api/models/switch', methods=['POST'])
        def switch_model():
            """Switch to a different trained model"""
            try:
                data = request.get_json()
                model_path = data.get('model_path')
                
                if not model_path or not os.path.exists(model_path):
                    return jsonify({
                        'success': False,
                        'error': 'Model path not found'
                    }), 400
                
                # Reload classifier with new model
                self.services['classifier'] = FashionClassifier(model_path)
                
                return jsonify({
                    'success': True,
                    'message': f'Switched to model: {model_path}',
                    'current_model': model_path
                })
                
            except Exception as e:
                logger.error(f"Model switch error: {str(e)}")
                return jsonify({'success': False, 'error': str(e)}), 500
    
    def _get_last_training_info(self):
        """Get information about the last training session"""
        try:
            metadata_dir = Path('models/metadata')
            if metadata_dir.exists():
                metadata_files = list(metadata_dir.glob('training_*.json'))
                if metadata_files:
                    latest_file = max(metadata_files, key=os.path.getmtime)
                    with open(latest_file, 'r') as f:
                        return json.load(f)
            return None
        except:
            return None
    
    def _setup_error_handlers(self):
        """Setup comprehensive error handling"""
        
        @self.app.errorhandler(404)
        def not_found(error):
            return jsonify({
                'success': False,
                'error': 'Endpoint not found',
                'available_endpoints': self._get_available_endpoints()
            }), 404
        
        @self.app.errorhandler(405)
        def method_not_allowed(error):
            return jsonify({
                'success': False,
                'error': 'Method not allowed'
            }), 405
        
        @self.app.errorhandler(413)
        def payload_too_large(error):
            return jsonify({
                'success': False,
                'error': f'File too large. Maximum size: {Config.MAX_CONTENT_LENGTH // (1024*1024)}MB'
            }), 413
        
        @self.app.errorhandler(500)
        def internal_server_error(error):
            logger.error(f"Internal server error: {str(error)}")
            return jsonify({
                'success': False,
                'error': 'Internal server error',
                'error_id': str(uuid.uuid4())
            }), 500
    
    def _setup_request_hooks(self):
        """Setup request hooks for monitoring"""
        
        @self.app.before_request
        def before_request():
            """Log and track requests"""
            self.request_count += 1
            request.start_time = datetime.now()
            
            # Log request
            logger.info(f"📝 {request.method} {request.path} - {request.remote_addr}")
        
        @self.app.after_request
        def after_request(response):
            """Log response and performance"""
            try:
                duration = (datetime.now() - request.start_time).total_seconds() * 1000
                
                # Log response
                logger.info(f"📤 {request.method} {request.path} - {response.status_code} - {duration:.2f}ms")
                
                # Track errors
                if response.status_code >= 400:
                    self.error_count += 1
                
                # Add performance headers
                response.headers['X-Response-Time'] = f"{duration:.2f}ms"
                response.headers['X-Request-ID'] = str(uuid.uuid4())
                
                return response
                
            except Exception as e:
                logger.error(f"After request hook error: {str(e)}")
                return response
    
    def _get_available_endpoints(self):
        """Get list of available API endpoints"""
        endpoints = []
        for rule in self.app.url_map.iter_rules():
            if rule.endpoint != 'static':
                endpoints.append({
                    'endpoint': rule.rule,
                    'methods': list(rule.methods - {'HEAD', 'OPTIONS'})
                })
        return endpoints
    
    def cleanup(self):
        """Cleanup resources on shutdown"""
        try:
            logger.info("🧹 Cleaning up resources...")
            
            # Stop model manager monitoring
            if 'model_manager' in self.services:
                self.services['model_manager'].stop_monitoring()
            
            # Save knowledge graph if modified
            if 'knowledge_service' in self.services:
                self.services['knowledge_service'].save_knowledge_graph()
            
            logger.info("✅ Cleanup completed")
            
        except Exception as e:
            logger.error(f"❌ Cleanup error: {str(e)}")

# Global backend instance
backend = FashionAIBackend()

def create_app():
    """Factory function for creating the Flask app"""
    return backend.create_app()

# Register cleanup on exit
atexit.register(backend.cleanup)

if __name__ == '__main__':
    try:
        # Create and run the application
        app = create_app()
        
        logger.info("=" * 60)
        logger.info("🚀 FASHION AI BACKEND WITH TRAINING STARTING")
        logger.info("=" * 60)
        logger.info(f"🏠 Host: {Config.HOST}")
        logger.info(f"🔌 Port: {Config.PORT}")
        logger.info(f"🐛 Debug: {Config.DEBUG}")
        logger.info(f"📊 Model Version: {Config.MODEL_VERSION}")
        logger.info(f"🧠 AI Services: {len(backend.services)} loaded")
        logger.info(f"🎓 Training: Enabled")
        logger.info("=" * 60)
        
        # Run the application
        app.run(
            host=Config.HOST,
            port=Config.PORT,
            debug=Config.DEBUG,
            threaded=True
        )
        
    except KeyboardInterrupt:
        logger.info("👋 Shutting down gracefully...")
    except Exception as e:
        logger.error(f"❌ Startup failed: {str(e)}")
        sys.exit(1)