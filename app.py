import os
import sys
import logging
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
from PIL import Image
import traceback
import uuid
import atexit
from pathlib import Path

# Import services
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
        # Remove API key - no authentication needed in AI backend
        
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
            
            # Enable CORS - TRUST requests from Node.js backend only
            CORS(self.app, resources={
                r"/api/*": {
                    "origins": ["http://localhost:8000"],  # Only Node.js backend
                    "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
                    "allow_headers": ["Content-Type", "Authorization"]
                },
                r"/health": {"origins": "*"}  # Health check can be accessed from anywhere
            })
            
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
            logger.error(f"❌ Blueprint registration error: {str(e)}")
            raise
    
    def _setup_routes(self):
        """Setup all API routes"""
        
        # REMOVED: No authentication needed - trust Node.js backend
        
        # Health check endpoint
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
                    'message': 'Fashion AI Backend with Training is healthy',
                    'timestamp': datetime.now().isoformat(),
                    'uptime_seconds': uptime,  # FIXED: was 'total_seconds'
                    'version': Config.MODEL_VERSION,
                    'services': service_status,
                    'request_count': self.request_count,  # FIXED: was 'total_request_count'
                    'error_count': self.error_count,  # FIXED: was 'self._error_count'
                    'error_rate': (self.error_count / max(self.request_count, 1)) * 100,
                    'features': {
                        'training_enabled': True,
                        'retraining_enabled': True,
                        'model_versioning': True,
                        'authentication': 'handled_by_nodejs'
                    }
                })
                
            except Exception as e:
                logger.error(f"Health check error: {str(e)}")
                self.error_count += 1
                return jsonify({'success': False, 'error': str(e)}), 500
        
        # Status endpoint
        @self.app.route('/api/status', methods=['GET'])
        def detailed_status():
            """Detailed system status including training capabilities"""
            try:
                status = self.services['model_manager'].get_status()
                
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
                logger.error(f"Status check error: {str(e)}")
                self.error_count += 1
                return jsonify({'success': False, 'error': str(e)}), 500
        
        # FIXED: Classification endpoint - removed auth, fixed field name
        @self.app.route('/api/classify', methods=['POST'])
        def classify_image():
            """Enhanced image classification with comprehensive analysis"""
            try:
                logger.info("🔍 Classification request received from Node.js backend")
                
                # FIXED: Check for 'image' field (not 'file') - matching Node.js backend
                if 'image' not in request.files:
                    logger.error("No 'image' field found in request.files")
                    logger.error(f"Available fields: {list(request.files.keys())}")
                    return jsonify({'success': False, 'error': 'No image file provided'}), 400
                
                file = request.files['image']  # FIXED: was 'file'
                if file.filename == '':
                    return jsonify({'success': False, 'error': 'No file selected'}), 400
                
                if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    return jsonify({'success': False, 'error': 'Unsupported file format. Use JPG or PNG'}), 400
                
                file.seek(0)
                raw_image = Image.open(file).convert('RGB')
                logger.info(f"Processing image: size={raw_image.size}, mode={raw_image.mode}")
                
                start_time = datetime.now()
                classification_result = self.services['classifier'].classify(raw_image)
                print("🔍 DEBUG: Raw classification result:")
                print(json.dumps(classification_result, indent=2, default=str))
                print("🔍 DEBUG: Predicted category:", classification_result.get('predicted_category'))
                print("🔍 DEBUG: Category (old key):", classification_result.get('category'))
                print("🔍 DEBUG: All predictions count:", len(classification_result.get('all_predictions', [])))
                
                # Reset file pointer for quality analysis
                file.seek(0)
                raw_image = Image.open(file).convert('RGB')
                quality_info = self.services['image_processor'].generate_image_quality_score(raw_image)
                
                attributes = {}
                if Config.ENABLE_ATTRIBUTE_ANALYSIS and 'attribute_analyzer' in self.services:
                    attributes = self.services['attribute_analyzer'].analyze(raw_image)
                
                processing_time = (datetime.now() - start_time).total_seconds() * 1000
                
                result = {
                    'success': True,
                    'classification': {
                        'predicted_class': classification_result.get('predicted_category', 'unknown'),
                        'confidence': classification_result.get('confidence', 0.0),
                        'all_predictions': classification_result.get('predictions', [])
                    },
                    'image_quality': quality_info,
                    'attributes': attributes,
                    'processing_time_ms': processing_time,
                    'model_version': Config.MODEL_VERSION,
                    'timestamp': datetime.now().isoformat()
                }
                
                logger.info(f"✅ Classification completed successfully in {processing_time:.2f}ms")
                return jsonify(result)
                
            except Exception as e:
                logger.error(f"❌ Classification error: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                self.error_count += 1
                return jsonify({'success': False, 'error': str(e)}), 500
        
        # FIXED: Similarity search endpoint - removed auth, fixed field name
        @self.app.route('/api/similarity/search', methods=['POST', 'OPTIONS'])
        def similarity_search():
            if request.method == 'OPTIONS':
                return jsonify({'success': True}), 200

            try:
                logger.info("🔍 Similarity search request received from Node.js backend")

                if not Config.ENABLE_SIMILARITY_SEARCH or 'similarity_engine' not in self.services:
                    return jsonify({'success': False, 'error': 'Similarity search is disabled'}), 503

                if not self.services['similarity_engine'].is_loaded():
                    return jsonify({
                        'success': False,
                        'error': 'Similarity database is empty or not loaded',
                        'database_size': 0
                    }), 404

                item_id = None
                category = None
                image = None
                limit = 5

                if request.is_json:
                    data = request.get_json()
                    item_id = data.get('itemId')
                    category = data.get('category', item_id)
                    limit = data.get('limit', 5)
                elif 'image' in request.files:  # FIXED: was 'file'
                    file = request.files['image']  # FIXED: was 'file'
                    if file.filename == '':
                        return jsonify({'success': False, 'error': 'No file selected'}), 400
                    if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                        return jsonify({'success': False, 'error': 'Unsupported file format'}), 400
                    file.seek(0)
                    image = Image.open(file).convert('RGB')
                    limit = int(request.form.get('limit', 5))
                    category = request.form.get('category')
                else:
                    return jsonify({'success': False, 'error': 'Invalid request: provide JSON or image file'}), 400

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

                logger.info(f"✅ Similarity search completed: {result['total']} items found")
                return jsonify(response), 200

            except Exception as e:
                logger.error(f"❌ Similarity search error: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                self.error_count += 1
                return jsonify({'success': False, 'error': str(e)}), 500

        # FIXED: Enhanced outfit generation endpoint with better debugging
        @self.app.route('/api/outfits/generate', methods=['POST'])
        def generate_outfits():
            try:
                logger.info("👗 Outfit generation request received from Node.js backend")

                data = request.get_json()
                if not data:
                    return jsonify({'success': False, 'error': 'No data provided'}), 400

                # Extract parameters
                wardrobe_items = data.get('wardrobe_items', [])
                user_id = data.get('user_id')
                occasion = data.get('occasion')
                season = data.get('season')
                count = data.get('count', 5)
                weather_context = data.get('weather_context', {})
                style_preferences = data.get('style_preferences', [])

                # Validate required fields
                if not all([wardrobe_items, user_id, occasion, season]):
                    missing_fields = []
                    if not wardrobe_items: missing_fields.append('wardrobe_items')
                    if not user_id: missing_fields.append('user_id')
                    if not occasion: missing_fields.append('occasion')
                    if not season: missing_fields.append('season')
                    
                    logger.error(f"Missing required fields: {missing_fields}")
                    return jsonify({
                        'success': False, 
                        'error': f'Missing required fields: {missing_fields}'
                    }), 400

                # ✅ ADDED: Debug wardrobe item categories
                categories_count = {}
                for item in wardrobe_items:
                    category = item.get('category', 'unknown')
                    categories_count[category] = categories_count.get(category, 0) + 1
                
                logger.info(f"📊 Wardrobe categories: {categories_count}")

                # ✅ ADDED: Check if we have minimum required categories
                category_mapping = {
                    'shirts_blouses': 'tops',
                    'tshirts_tops': 'tops',
                    'sweaters': 'tops',
                    'dresses': 'dresses',
                    'pants_jeans': 'bottoms',
                    'shorts': 'bottoms',
                    'skirts': 'bottoms',
                    'shoes_sneakers': 'shoes',
                    'shoes_formal': 'shoes',
                    'jackets_coats': 'outerwear',
                    'bags_accessories': 'accessories'
                }
                
                available_categories = set()
                for item in wardrobe_items:
                    category = item.get('category', 'unknown')
                    mapped_category = category_mapping.get(category, 'accessories')
                    available_categories.add(mapped_category)
                
                logger.info(f"📊 Available mapped categories: {list(available_categories)}")
                
                # Check for required categories based on occasion
                required_categories = ['tops', 'bottoms', 'shoes']  # For most occasions
                missing_categories = set(required_categories) - available_categories
                
                if missing_categories:
                    logger.warning(f"⚠️ Missing required categories: {list(missing_categories)}")
                    # ✅ ADDED: Return helpful error message instead of 0 outfits
                    return jsonify({
                        'success': True,  # Don't fail the request
                        'outfits': [],
                        'total_generated': 0,
                        'algorithm_info': {
                            'error': f'Cannot create outfits: missing {list(missing_categories)}',
                            'available_categories': list(available_categories),
                            'required_categories': required_categories,
                            'suggestions': [
                                f"Add some {cat} items to your wardrobe" for cat in missing_categories
                            ]
                        },
                        'processing_time_ms': 0,
                        'timestamp': datetime.now().isoformat()
                    })

                logger.info(f"Processing outfit generation: user_id={user_id}, items={len(wardrobe_items)}, occasion={occasion}, season={season}")

                start_time = datetime.now()
                
                # Call the recommendation engine
                outfits_result = self.services['recommendation_engine'].generate_outfits(
                    user_id=user_id,
                    wardrobe_items=wardrobe_items,
                    occasion=occasion,
                    season=season,
                    weather_context=weather_context,
                    style_preferences=style_preferences,
                    count=count
                )
                
                processing_time = (datetime.now() - start_time).total_seconds() * 1000

                # ✅ ADDED: Better debugging of results
                outfits_list = outfits_result.get('outfits', [])
                logger.info(f"✅ Outfit generation completed: {len(outfits_list)} outfits generated in {processing_time:.2f}ms")
                
                if len(outfits_list) == 0:
                    logger.warning(f"⚠️ No outfits generated. Algorithm info: {outfits_result.get('algorithm_info', {})}")

                # ✅ FIXED: Ensure response structure matches frontend expectations
                response = {
                    'success': True,
                    'outfits': outfits_list,
                    'total_generated': outfits_result.get('total', len(outfits_list)),
                    'algorithm_info': outfits_result.get('algorithm_info', {}),
                    'processing_time_ms': processing_time,
                    'timestamp': datetime.now().isoformat()
                }

                # ✅ ADDED: Transform outfits to match frontend expectations
                if outfits_list:
                    for outfit in response['outfits']:
                        # Ensure required fields exist
                        outfit.setdefault('id', outfit.get('_id', str(uuid.uuid4())))
                        outfit.setdefault('name', f"{occasion.title()} Outfit")
                        outfit.setdefault('score', int(outfit.get('overall_score', 0.8) * 100))
                        outfit.setdefault('tags', [occasion, season])
                        outfit.setdefault('items', outfit.get('item_ids', []))
                        outfit.setdefault('explanation', outfit.get('explanation', []))
                        outfit.setdefault('weatherContext', weather_context)

                return jsonify(response)
                
            except Exception as e:
                logger.error(f"❌ Outfit generation error: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                self.error_count += 1
                return jsonify({'success': False, 'error': str(e)}), 500


        @self.app.route('/api/compatibility', methods=['POST'])
        def check_compatibility():
            try:
                logger.info("🤝 Compatibility check request received from Node.js backend")

                data = request.get_json()
                if not data:
                    return jsonify({'success': False, 'error': 'No data provided'}), 400

                item1_id = data.get('item1_id')
                item2_id = data.get('item2_id')
                context = data.get('context')

                if not all([item1_id, item2_id]):
                    return jsonify({'success': False, 'error': 'Missing item IDs'}), 400

                start_time = datetime.now()
                compatibility = self.services['compatibility_engine'].check_compatibility(
                    item1_id=item1_id,
                    item2_id=item2_id,
                    context=context
                )
                processing_time = (datetime.now() - start_time).total_seconds() * 1000

                return jsonify({
                    'success': True,
                    'compatibility': compatibility,
                    'processing_time_ms': processing_time,
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"❌ Compatibility check error: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                self.error_count += 1
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/style/recommendations', methods=['POST'])
        def style_recommendations():
            try:
                logger.info("💡 Style recommendations request received from Node.js backend")

                data = request.get_json()
                if not data:
                    return jsonify({'success': False, 'error': 'No data provided'}), 400

                base_item_id = data.get('base_item_id')
                context = data.get('context')
                limit = data.get('limit', 5)

                if not base_item_id:
                    return jsonify({'success': False, 'error': 'Missing base item ID'}), 400

                start_time = datetime.now()
                recommendations = self.services['recommendation_engine'].get_style_recommendations(
                    base_item_id=base_item_id,
                    context=context,
                    limit=limit
                )
                processing_time = (datetime.now() - start_time).total_seconds() * 1000

                return jsonify({
                    'success': True,
                    'recommendations': recommendations,
                    'total': len(recommendations),
                    'processing_time_ms': processing_time,
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"❌ Style recommendations error: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                self.error_count += 1
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/attributes/analyze', methods=['POST'])
        def analyze_attributes():
            try:
                logger.info("🏷️ Attribute analysis request received from Node.js backend")

                if 'image' not in request.files:  # FIXED: was 'file'
                    return jsonify({'success': False, 'error': 'No image file provided'}), 400

                file = request.files['image']  # FIXED: was 'file'
                if file.filename == '':
                    return jsonify({'success': False, 'error': 'No file selected'}), 400

                if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    return jsonify({'success': False, 'error': 'Unsupported file format'}), 400

                file.seek(0)
                raw_image = Image.open(file).convert('RGB')
                logger.info(f"Analyzing attributes for image: size={raw_image.size}")

                start_time = datetime.now()
                attributes = self.services['attribute_analyzer'].analyze(raw_image)
                processing_time = (datetime.now() - start_time).total_seconds() * 1000

                return jsonify({
                    'success': True,
                    'attributes': attributes,
                    'processing_time_ms': processing_time,
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"❌ Attribute analysis error: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                self.error_count += 1
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/models/switch', methods=['POST'])
        def switch_model():
            try:
                logger.info("🔄 Model switch request received from Node.js backend")

                data = request.get_json()
                model_path = data.get('model_path')
                
                if not model_path or not os.path.exists(model_path):
                    return jsonify({
                        'success': False,
                        'error': 'Model path not found'
                    }), 400
                
                self.services['classifier'] = FashionClassifier(model_path)
                
                return jsonify({
                    'success': True,
                    'message': f'Switched to model: {model_path}',
                    'current_model': model_path,
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"❌ Model switch error: {str(e)}")
                self.error_count += 1
                return jsonify({'success': False, 'error': str(e)}), 500
    
    def _get_last_training_info(self):
        """Retrieve the latest training metadata"""
        try:
            metadata_dir = Path('models/metadata')
            if metadata_dir.exists():
                metadata_files = list(metadata_dir.glob('training_*.json'))
                if metadata_files:
                    latest_file = max(metadata_files, key=os.path.getmtime)
                    with open(latest_file, 'r') as f:
                        return json.load(f)
            return None
        except Exception as e:
            logger.error(f"Error retrieving training info: {str(e)}")
            return None
    
    def _setup_error_handlers(self):
        """Setup custom error handlers"""
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
            self.error_count += 1
            return jsonify({
                'success': False,
                'error': 'Internal server error',
                'error_id': str(uuid.uuid4())
            }), 500
    
    def _setup_request_hooks(self):
        """Setup request logging hooks"""
        @self.app.before_request
        def before_request():
            self.request_count += 1
            request.start_time = datetime.now()
            logger.info(f"📝 {request.method} {request.path} - {request.remote_addr}")
        
        @self.app.after_request
        def after_request(response):
            try:
                duration = (datetime.now() - request.start_time).total_seconds() * 1000
                logger.info(f"📤 {request.method} {request.path} - {response.status_code} - {duration:.2f}ms")
                if response.status_code >= 400:
                    self.error_count += 1
                response.headers['X-Response-Time'] = f"{duration:.2f}ms"
                response.headers['X-Request-ID'] = str(uuid.uuid4())
                return response
            except Exception as e:
                logger.error(f"After request hook error: {str(e)}")
                return response
    
    def _get_available_endpoints(self):
        """List all available API endpoints"""
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
            if 'model_manager' in self.services:
                self.services['model_manager'].stop_monitoring()
            if 'knowledge_service' in self.services:
                self.services['knowledge_service'].save_knowledge_graph()
            logger.info("✅ Cleanup completed")
        except Exception as e:
            logger.error(f"❌ Cleanup error: {str(e)}")

# Global backend instance
backend = FashionAIBackend()

def create_app():
    return backend.create_app()

atexit.register(backend.cleanup)

if __name__ == '__main__':
    try:
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
        logger.info(f"🔐 Authentication: Handled by Node.js Backend")
        logger.info("=" * 60)
        
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
