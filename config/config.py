import os
import secrets
from datetime import timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # Flask configuration
    SECRET_KEY = os.getenv('SECRET_KEY', secrets.token_hex(32))
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    
    # Session configuration
    SESSION_TYPE = 'filesystem'
    SESSION_PERMANENT = True
    PERMANENT_SESSION_LIFETIME = timedelta(days=31)
    SESSION_COOKIE_SECURE = os.getenv('PROD', 'False').lower() == 'true'
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    
    # Upload configuration
    UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads')
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    MAX_CONTENT_LENGTH = int(os.getenv('MAX_CONTENT_LENGTH', 16 * 1024 * 1024))
    
    # MongoDB configuration
    MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
    DB_NAME = os.getenv('DB_NAME', 'wardrobe_assistant')
    
    # Model configuration
    MODEL_PATH = os.getenv('MODEL_PATH', 'real_fashion_classifier_final.keras')
    
    # Fallback JSON file (if MongoDB unavailable)
    WARDROBE_DB = 'wardrobe_db.json'
    
    # Weather and season mappings (moved from app.py)
    @staticmethod
    def get_category_details():
        return {
            'top': {
                'layer': 'upper',
                'coverage': 'partial',
                'formality_range': [1, 8],
            },
            'bottom': {
                'layer': 'lower',
                'coverage': 'partial',
                'formality_range': [1, 9],
            },
            'full': {
                'layer': 'full',
                'coverage': 'full',
                'formality_range': [3, 10],
            },
            'outer': {
                'layer': 'upper',
                'coverage': 'partial',
                'is_outer': True,
                'formality_range': [2, 9],
            },
            'footwear': {
                'layer': 'foot',
                'coverage': 'partial',
                'formality_range': [1, 10],
            },
            'accessory': {
                'layer': 'accessory',
                'coverage': 'minimal',
                'formality_range': [1, 10],
            }
        }
    
    @staticmethod
    def get_weather_outfit_mapping():
        return {
            'clear': {
                'hot': ['lightweight', 'breathable', 'short sleeves', 'shorts'],
                'warm': ['light layers', 'short sleeves', 'light pants'],
                'mild': ['light layers', 'long sleeves', 'light jacket'],
                'cool': ['layers', 'light jacket', 'long pants'],
                'cold': ['heavy layers', 'coat', 'long pants', 'scarf']
            },
            'rain': {
                'warm': ['water resistant', 'light rain jacket', 'covered shoes'],
                'mild': ['water resistant', 'rain jacket', 'covered shoes'],
                'cool': ['water resistant', 'rain jacket', 'layers', 'covered shoes'],
                'cold': ['waterproof', 'rain jacket', 'heavy layers', 'waterproof boots']
            },
            'snow': {
                'cold': ['waterproof', 'insulated', 'heavy coat', 'boots', 'hat', 'gloves']
            }
        }
