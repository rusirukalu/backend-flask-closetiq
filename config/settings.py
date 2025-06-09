# config/settings.py - System Configuration Management
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration management for Fashion AI Backend"""
    
    # Base directories
    BASE_DIR = Path(__file__).parent.parent
    MODELS_DIR = BASE_DIR / 'models'
    UPLOADS_DIR = BASE_DIR / 'uploads'
    LOGS_DIR = BASE_DIR / 'logs'
    
    @classmethod
    def create_dirs(cls):
        """Create necessary directories"""
        cls.MODELS_DIR.mkdir(exist_ok=True)
        cls.UPLOADS_DIR.mkdir(exist_ok=True)
        cls.LOGS_DIR.mkdir(exist_ok=True)
    
    # Server configuration
    PORT = int(os.environ.get('PORT', 5002))
    HOST = os.environ.get('HOST', '0.0.0.0')
    DEBUG = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    
    # MongoDB URI - This was missing!
    MONGODB_URI = os.environ.get('MONGODB_URI')
    if not MONGODB_URI:
        raise ValueError("MONGODB_URI environment variable is required")
    
    # Model paths
    BASE_MODEL_PATH = os.environ.get('BASE_MODEL_PATH', str(MODELS_DIR / 'final_enhanced_model.keras'))
    SIMILARITY_DB_PATH = os.environ.get('SIMILARITY_DB_PATH', str(MODELS_DIR / 'fashion_similarity_database.json'))
    KNOWLEDGE_GRAPH_PATH = os.environ.get('KNOWLEDGE_GRAPH_PATH', str(MODELS_DIR / 'fashion_knowledge_graph.pkl'))
    
    # AI Configuration
    CONFIDENCE_THRESHOLD = float(os.environ.get('CONFIDENCE_THRESHOLD', 0.7))
    MAX_BATCH_SIZE = int(os.environ.get('MAX_BATCH_SIZE', 10))
    MODEL_VERSION = os.environ.get('MODEL_VERSION', '1.0.0')
    
    # Performance settings
    MAX_MODELS_IN_MEMORY = int(os.environ.get('MAX_MODELS_IN_MEMORY', 3))
    MEMORY_THRESHOLD = float(os.environ.get('MEMORY_THRESHOLD', 0.8))
    PREDICTION_TIMEOUT = int(os.environ.get('PREDICTION_TIMEOUT', 30))
    
    # File upload settings
    MAX_CONTENT_LENGTH = int(os.environ.get('MAX_CONTENT_LENGTH', 16 * 1024 * 1024))  # 16MB
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
    
    # External service URLs
    NODEJS_BACKEND_URL = os.environ.get('NODEJS_BACKEND_URL', 'http://localhost:8000')
    
    # Logging configuration
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Cache settings
    ENABLE_CACHING = os.environ.get('ENABLE_CACHING', 'True').lower() == 'true'
    CACHE_TTL = int(os.environ.get('CACHE_TTL', 3600))  # 1 hour
    
    # Security settings
    API_KEY_REQUIRED = os.environ.get('API_KEY_REQUIRED', 'False').lower() == 'true'
    API_KEY = os.environ.get('API_KEY', 'fashion-ai-secret-key')
    
    # Feature flags
    ENABLE_SIMILARITY_SEARCH = os.environ.get('ENABLE_SIMILARITY_SEARCH', 'True').lower() == 'true'
    ENABLE_KNOWLEDGE_GRAPH = os.environ.get('ENABLE_KNOWLEDGE_GRAPH', 'True').lower() == 'true'
    ENABLE_ATTRIBUTE_ANALYSIS = os.environ.get('ENABLE_ATTRIBUTE_ANALYSIS', 'True').lower() == 'true'
    ENABLE_PERFORMANCE_MONITORING = os.environ.get('ENABLE_PERFORMANCE_MONITORING', 'True').lower() == 'true'
    
    @classmethod
    def validate_config(cls):
        """Validate configuration settings"""
        errors = []
        
        # Check MongoDB URI
        if not cls.MONGODB_URI:
            errors.append("MONGODB_URI environment variable is required")
        
        # Check required files
        if not os.path.exists(cls.BASE_MODEL_PATH):
            errors.append(f"Base model not found: {cls.BASE_MODEL_PATH}")
        
        # Validate numeric ranges
        if not 0 < cls.CONFIDENCE_THRESHOLD <= 1:
            errors.append("CONFIDENCE_THRESHOLD must be between 0 and 1")
        
        if not 0 < cls.MEMORY_THRESHOLD <= 1:
            errors.append("MEMORY_THRESHOLD must be between 0 and 1")
        
        if cls.MAX_BATCH_SIZE <= 0:
            errors.append("MAX_BATCH_SIZE must be positive")
        
        return errors

# Create directories when module is loaded
Config.create_dirs()
