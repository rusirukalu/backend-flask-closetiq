# api/training.py - Training API Endpoints
from flask import Blueprint, request, jsonify
import threading
from datetime import datetime
import os

from services.training_service import TrainingOrchestrator

training_bp = Blueprint('training', __name__)

# Global training state
training_status = {
    'is_training': False,
    'current_job': None,
    'progress': 0,
    'message': 'Ready',
    'last_model_path': None
}

@training_bp.route('/start', methods=['POST'])
def start_training():
    """Start model training"""
    try:
        if training_status['is_training']:
            return jsonify({
                'success': False,
                'error': 'Training already in progress'
            }), 400
        
        data = request.get_json() or {}
        
        # Start training in background
        training_thread = threading.Thread(
            target=_run_training,
            args=(data,)
        )
        training_thread.start()
        
        return jsonify({
            'success': True,
            'message': 'Training started',
            'job_id': training_status['current_job']
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@training_bp.route('/retrain', methods=['POST'])
def start_retraining():
    """Start model retraining"""
    try:
        if training_status['is_training']:
            return jsonify({
                'success': False,
                'error': 'Training already in progress'
            }), 400
        
        data = request.get_json()
        if not data or 'base_model_path' not in data or 'new_data_path' not in data:
            return jsonify({
                'success': False,
                'error': 'base_model_path and new_data_path are required'
            }), 400
        
        # Start retraining in background
        retraining_thread = threading.Thread(
            target=_run_retraining,
            args=(data,)
        )
        retraining_thread.start()
        
        return jsonify({
            'success': True,
            'message': 'Retraining started',
            'job_id': training_status['current_job']
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@training_bp.route('/status', methods=['GET'])
def get_training_status():
    """Get current training status"""
    return jsonify({
        'success': True,
        'status': training_status
    })

@training_bp.route('/models', methods=['GET'])
def list_models():
    """List available trained models"""
    try:
        models_dir = Path('models/trained')
        model_files = list(models_dir.glob('*.keras'))
        
        models = []
        for model_file in model_files:
            stat = model_file.stat()
            models.append({
                'name': model_file.name,
                'path': str(model_file),
                'size_mb': stat.st_size / (1024 * 1024),
                'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
            })
        
        return jsonify({
            'success': True,
            'models': sorted(models, key=lambda x: x['modified'], reverse=True)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def _run_training(params):
    """Run training in background"""
    global training_status
    
    try:
        training_status.update({
            'is_training': True,
            'current_job': f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'message': 'Starting training pipeline...',
            'progress': 0
        })
        
        orchestrator = TrainingOrchestrator()
        model_path = orchestrator.train_new_model(params)
        
        training_status.update({
            'message': f'Training completed: {model_path}',
            'progress': 100,
            'last_model_path': model_path
        })
        
    except Exception as e:
        training_status['message'] = f'Training failed: {str(e)}'
    finally:
        training_status['is_training'] = False

def _run_retraining(params):
    """Run retraining in background"""
    global training_status
    
    try:
        training_status.update({
            'is_training': True,
            'current_job': f"retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'message': 'Starting retraining...',
            'progress': 0
        })
        
        orchestrator = TrainingOrchestrator()
        model_path = orchestrator.retrain_model(
            params['base_model_path'],
            params['new_data_path']
        )
        
        training_status.update({
            'message': f'Retraining completed: {model_path}',
            'progress': 100,
            'last_model_path': model_path
        })
        
    except Exception as e:
        training_status['message'] = f'Retraining failed: {str(e)}'
    finally:
        training_status['is_training'] = False
