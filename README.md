# ClosetIQ Backend - Flask Fashion AI API

A comprehensive Flask-based backend API for fashion classification, style compatibility analysis, and intelligent clothing recommendations. ClosetIQ leverages advanced machine learning models to provide fashion insights through multiple specialized services.

## **Features**

- **Fashion Classification**: Multi-label attribute prediction for clothing items
- **Style Compatibility**: Intelligent outfit matching and compatibility scoring
- **Similarity Engine**: Fashion item similarity analysis and recommendations
- **Knowledge Graph**: Comprehensive fashion relationship mapping
- **Object Detection**: Clothing item localization and identification
- **Training Pipeline**: Complete model training and retraining workflows

## **Project Structure**

```
backend-flask-closetiq/
├── api/                    # API endpoints and routing
├── config/                 # Configuration files and settings
├── logs/                   # Application logs
├── models/                 # ML models, weights, and metadata
│   ├── trained/           # Production-ready models
│   ├── checkpoints/       # Training checkpoints
│   └── metadata/          # Model performance reports
├── notebooks/             # Jupyter notebooks for analysis
├── scripts/               # Training and utility scripts
├── services/              # Core business logic services
├── tests/                 # Test files
├── uploads/               # File upload directory
├── utils/                 # Utility functions and helpers
├── app.py                 # Main Flask application
└── requirements.txt       # Python dependencies
```

## **Prerequisites**

- **Python 3.12** or higher
- **Flask** web framework
- **TensorFlow/Keras** for machine learning models
- **NumPy** and **scikit-learn** for data processing
- **Kaggle API** credentials (for dataset access)

## **Installation \& Setup**

### **1. Clone the Repository**

```bash
git clone <repository-url>
cd backend-flask-closetiq
```

### **2. Create Virtual Environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### **3. Install Dependencies**

```bash
pip install -r requirements.txt
```

### **4. Environment Configuration**

Create a `.env` file in the root directory:

```env
FLASK_APP=app.py
FLASK_ENV=development
FLASK_DEBUG=True
SECRET_KEY=your-secret-key-here
UPLOAD_FOLDER=uploads
MODEL_PATH=models/trained
```

### **5. Configure Kaggle API**

Place your `kaggle.json` credentials file in the `config/` directory:

```bash
cp path/to/your/kaggle.json config/kaggle.json
chmod 600 config/kaggle.json
```

### **6. Initialize Application**

```bash
# Create necessary directories
mkdir -p logs uploads models/checkpoints models/exports

# Run the Flask application
flask run
```

The API will be available at `http://localhost:5000`

## **API Endpoints**

### **Classification Service**

- `POST /api/classify` - Classify fashion items and extract attributes
- `POST /api/attributes` - Get detailed attribute predictions

### **Recommendation Service**

- `POST /api/recommend` - Get fashion recommendations
- `POST /api/similarity` - Find similar fashion items

### **Compatibility Service**

- `POST /api/compatibility` - Check style compatibility between items
- `POST /api/outfit-score` - Score complete outfit combinations

### **Training Service**

- `POST /api/train` - Trigger model training
- `GET /api/training-status` - Check training progress

## **Model Training**

The project includes a comprehensive training pipeline with multiple steps:

### **Quick Training**

```bash
# Basic model training
python scripts/train_model.py

# Retrain existing model
python scripts/retrain_model.py
```

### **Advanced Training Pipeline**

```bash
# Step-by-step training process
python scripts/training/step1_advanced_attribute_classifier.py
python scripts/training/step2_data_improvement.py
python scripts/training/step3_multilabel_attribute_prediction.py
# ... continue with additional steps as needed
```

### **Model Analysis**

```bash
# Analyze model performance
python scripts/analysis/model_analysis.py

# Visual categorization analysis
python scripts/analysis/visual_categorizer.py
```

## **Configuration**

Configuration files are located in the `config/` directory:

- `settings.py` - Main application settings
- `model_config.py` - ML model configurations
- `*_config.json` - Service-specific configurations

## **Testing**

Run the test suite:

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_integration.py

# Run with coverage
python -m pytest tests/ --cov=.
```

## **Services Architecture**

### **Core Services**

- **Classification Service**: Handles fashion item classification and attribute extraction
- **Similarity Service**: Manages fashion similarity calculations and database operations
- **Compatibility Service**: Analyzes style compatibility between fashion items
- **Knowledge Service**: Manages the fashion knowledge graph
- **Recommendation Service**: Provides personalized fashion recommendations
- **Training Service**: Handles model training and retraining workflows

### **Utility Modules**

- **Image Processor**: Image preprocessing and augmentation
- **Model Manager**: Model loading, saving, and version management
- **Data Utils**: Data processing and transformation utilities

## **Model Information**

The system uses multiple specialized models:

- **Fashion Classifier**: Multi-label classification for clothing attributes
- **Similarity Engine**: Vector-based fashion similarity matching
- **Compatibility Scorer**: Style compatibility prediction
- **Knowledge Graph**: Relationship mapping between fashion concepts

Models are stored in the `models/trained/` directory and automatically loaded on application startup.

## **Development**

### **Adding New Features**

1. Create service modules in `services/`
2. Add API endpoints in `api/`
3. Update configuration in `config/`
4. Add tests in `tests/`

### **Model Development**

1. Use notebooks in `notebooks/` for experimentation
2. Implement training scripts in `scripts/training/`
3. Add analysis tools in `scripts/analysis/`
4. Update model configurations as needed

## **Deployment**

### **Production Setup**

```bash
# Set production environment
export FLASK_ENV=production
export FLASK_DEBUG=False

# Use production WSGI server
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### **Docker Deployment**

```bash
# Build container
docker build -t closetiq-backend .

# Run container
docker run -p 5000:5000 closetiq-backend
```

## **Contributing**

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## **License**

This project is licensed under the MIT License - see the LICENSE file for details.

## **Support**

For questions and support, please open an issue in the repository or contact the development team.
