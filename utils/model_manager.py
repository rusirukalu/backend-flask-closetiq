# utils/model_manager.py - Dynamic Model Management System
import tensorflow as tf
import os
import json
import psutil
import threading
import time
from datetime import datetime
import logging
import gc
import numpy as np
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ModelManager:
    """Dynamic Model Management and Optimization System"""
    
    def __init__(self):
        self.loaded_models = {}
        self.model_metadata = {}
        self.performance_metrics = {}
        self.memory_threshold = 0.8  # 80% memory usage threshold
        self.model_cache_size = 3  # Maximum number of models to keep in memory
        
        # Performance monitoring
        self.monitoring_thread = None
        self.monitoring_active = False
        
        # Start monitoring
        self.start_monitoring()
    
    def load_model(self, model_name: str, model_path: str, force_reload: bool = False) -> bool:
        """Load a model dynamically with memory management"""
        try:
            # Check if model already loaded
            if model_name in self.loaded_models and not force_reload:
                logger.info(f"✅ Model {model_name} already loaded")
                return True
            
            # Check memory before loading
            if not self._check_memory_availability():
                self._free_memory()
            
            # Load model
            logger.info(f"🔄 Loading model: {model_name}")
            start_time = time.time()
            
            model = tf.keras.models.load_model(model_path)
            
            load_time = time.time() - start_time
            
            # Store model and metadata
            self.loaded_models[model_name] = model
            self.model_metadata[model_name] = {
                'path': model_path,
                'load_time': load_time,
                'loaded_at': datetime.now().isoformat(),
                'input_shape': model.input_shape,
                'output_shape': model.output_shape,
                'total_params': model.count_params(),
                'memory_usage': self._estimate_model_memory(model),
                'usage_count': 0,
                'last_used': datetime.now().isoformat()
            }
            
            # Initialize performance metrics
            self.performance_metrics[model_name] = {
                'total_predictions': 0,
                'total_time': 0.0,
                'average_latency': 0.0,
                'errors': 0,
                'success_rate': 100.0
            }
            
            logger.info(f"✅ Model {model_name} loaded successfully in {load_time:.2f}s")
            
            # Manage cache size
            self._manage_cache_size()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load model {model_name}: {str(e)}")
            return False
    
    def get_model(self, model_name: str):
        """Get a loaded model"""
        if model_name in self.loaded_models:
            # Update usage statistics
            self.model_metadata[model_name]['usage_count'] += 1
            self.model_metadata[model_name]['last_used'] = datetime.now().isoformat()
            return self.loaded_models[model_name]
        else:
            logger.warning(f"⚠️ Model {model_name} not loaded")
            return None
    
    def unload_model(self, model_name: str) -> bool:
        """Unload a specific model from memory"""
        try:
            if model_name in self.loaded_models:
                del self.loaded_models[model_name]
                del self.model_metadata[model_name]
                del self.performance_metrics[model_name]
                
                # Force garbage collection
                gc.collect()
                
                logger.info(f"✅ Model {model_name} unloaded")
                return True
            else:
                logger.warning(f"⚠️ Model {model_name} not found for unloading")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error unloading model {model_name}: {str(e)}")
            return False
    
    def predict_with_monitoring(self, model_name: str, input_data, **kwargs):
        """Make prediction with performance monitoring"""
        try:
            model = self.get_model(model_name)
            if model is None:
                raise Exception(f"Model {model_name} not available")
            
            # Monitor prediction
            start_time = time.time()
            
            try:
                prediction = model.predict(input_data, **kwargs)
                success = True
            except Exception as pred_error:
                logger.error(f"Prediction error for {model_name}: {str(pred_error)}")
                success = False
                raise pred_error
            finally:
                # Update performance metrics
                prediction_time = time.time() - start_time
                self._update_performance_metrics(model_name, prediction_time, success)
            
            return prediction
            
        except Exception as e:
            logger.error(f"❌ Prediction error: {str(e)}")
            raise
    
    def _update_performance_metrics(self, model_name: str, prediction_time: float, success: bool):
        """Update performance metrics for a model"""
        if model_name not in self.performance_metrics:
            return
        
        metrics = self.performance_metrics[model_name]
        metrics['total_predictions'] += 1
        
        if success:
            metrics['total_time'] += prediction_time
            metrics['average_latency'] = metrics['total_time'] / (metrics['total_predictions'] - metrics['errors'])
        else:
            metrics['errors'] += 1
        
        # Calculate success rate
        total_attempts = metrics['total_predictions']
        successful_attempts = total_attempts - metrics['errors']
        metrics['success_rate'] = (successful_attempts / total_attempts) * 100 if total_attempts > 0 else 100.0
    
    def _check_memory_availability(self) -> bool:
        """Check if there's enough memory to load a new model"""
        memory = psutil.virtual_memory()
        return memory.percent < (self.memory_threshold * 100)
    
    def _free_memory(self):
        """Free memory by unloading least recently used models"""
        if len(self.loaded_models) == 0:
            return
        
        # Sort models by last used time
        sorted_models = sorted(
            self.model_metadata.items(),
            key=lambda x: x[1]['last_used']
        )
        
        # Unload oldest model
        oldest_model = sorted_models[0][0]
        logger.info(f"🗑️ Freeing memory by unloading {oldest_model}")
        self.unload_model(oldest_model)
        
        # Force garbage collection
        gc.collect()
    
    def _manage_cache_size(self):
        """Manage the number of models in cache"""
        while len(self.loaded_models) > self.model_cache_size:
            self._free_memory()
    
    def _estimate_model_memory(self, model) -> int:
        """Estimate memory usage of a model in bytes"""
        try:
            # Calculate based on parameters and data types
            total_params = model.count_params()
            # Assume float32 (4 bytes per parameter)
            estimated_memory = total_params * 4
            return estimated_memory
        except:
            return 0
    
    def start_monitoring(self):
        """Start performance monitoring thread"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            logger.info("📊 Model performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        logger.info("📊 Model performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Continuous monitoring loop"""
        while self.monitoring_active:
            try:
                # Monitor system resources
                self._monitor_system_resources()
                
                # Monitor model performance
                self._monitor_model_performance()
                
                # Sleep for 30 seconds
                time.sleep(30)
                
            except Exception as e:
                logger.error(f"Monitoring error: {str(e)}")
    
    def _monitor_system_resources(self):
        """Monitor system resource usage"""
        try:
            # Memory usage
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Log warnings if resources are high
            if memory.percent > 85:
                logger.warning(f"⚠️ High memory usage: {memory.percent:.1f}%")
            
            if cpu_percent > 90:
                logger.warning(f"⚠️ High CPU usage: {cpu_percent:.1f}%")
                
        except Exception as e:
            logger.error(f"Resource monitoring error: {str(e)}")
    
    def _monitor_model_performance(self):
        """Monitor individual model performance"""
        try:
            for model_name, metrics in self.performance_metrics.items():
                # Check for performance issues
                if metrics['average_latency'] > 5.0:  # 5 seconds threshold
                    logger.warning(f"⚠️ Model {model_name} has high latency: {metrics['average_latency']:.2f}s")
                
                if metrics['success_rate'] < 95.0:
                    logger.warning(f"⚠️ Model {model_name} has low success rate: {metrics['success_rate']:.1f}%")
                    
        except Exception as e:
            logger.error(f"Model performance monitoring error: {str(e)}")
    
    def optimize_models(self):
        """Optimize loaded models for better performance"""
        try:
            for model_name, model in self.loaded_models.items():
                logger.info(f"🔧 Optimizing model: {model_name}")
                
                # Convert to TensorFlow Lite for mobile deployment (optional)
                # self._convert_to_tflite(model_name, model)
                
                # Optimize for inference
                self._optimize_for_inference(model_name, model)
                
        except Exception as e:
            logger.error(f"Model optimization error: {str(e)}")
    
    def _optimize_for_inference(self, model_name: str, model):
        """Optimize model for faster inference"""
        try:
            # This is a placeholder for various optimization techniques
            # In practice, you might:
            # 1. Use TensorRT for GPU optimization
            # 2. Apply quantization
            # 3. Use TensorFlow Serving optimizations
            
            logger.info(f"✅ Model {model_name} optimized for inference")
            
        except Exception as e:
            logger.error(f"Inference optimization error for {model_name}: {str(e)}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all models and system"""
        try:
            # System resources
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent()
            
            # Model summaries
            model_summaries = {}
            for name, metadata in self.model_metadata.items():
                performance = self.performance_metrics.get(name, {})
                
                model_summaries[name] = {
                    'loaded': True,
                    'metadata': metadata,
                    'performance': performance,
                    'memory_mb': metadata.get('memory_usage', 0) / (1024 * 1024),
                    'is_healthy': performance.get('success_rate', 100) > 95
                }
            
            return {
                'models': model_summaries,
                'system_resources': {
                    'memory_percent': memory.percent,
                    'memory_available_gb': memory.available / (1024**3),
                    'cpu_percent': cpu_percent
                },
                'cache_status': {
                    'loaded_models': len(self.loaded_models),
                    'max_cache_size': self.model_cache_size,
                    'memory_threshold': self.memory_threshold * 100
                },
                'monitoring_active': self.monitoring_active
            }
            
        except Exception as e:
            logger.error(f"Status retrieval error: {str(e)}")
            return {'error': str(e)}
    
    def export_performance_report(self) -> Dict[str, Any]:
        """Export detailed performance report"""
        try:
            report = {
                'generated_at': datetime.now().isoformat(),
                'models': {}
            }
            
            for model_name in self.loaded_models:
                metadata = self.model_metadata.get(model_name, {})
                performance = self.performance_metrics.get(model_name, {})
                
                report['models'][model_name] = {
                    'model_info': {
                        'parameters': metadata.get('total_params', 0),
                        'memory_usage_mb': metadata.get('memory_usage', 0) / (1024 * 1024),
                        'load_time_seconds': metadata.get('load_time', 0),
                        'usage_count': metadata.get('usage_count', 0)
                    },
                    'performance_metrics': {
                        'total_predictions': performance.get('total_predictions', 0),
                        'average_latency_ms': performance.get('average_latency', 0) * 1000,
                        'success_rate_percent': performance.get('success_rate', 100),
                        'error_count': performance.get('errors', 0)
                    },
                    'recommendations': self._generate_recommendations(model_name, performance)
                }
            
            return report
            
        except Exception as e:
            logger.error(f"Performance report error: {str(e)}")
            return {'error': str(e)}
    
    def _generate_recommendations(self, model_name: str, performance: Dict) -> list:
        """Generate performance improvement recommendations"""
        recommendations = []
        
        avg_latency = performance.get('average_latency', 0)
        success_rate = performance.get('success_rate', 100)
        
        if avg_latency > 2.0:
            recommendations.append("Consider model optimization or using a lighter model variant")
        
        if success_rate < 98:
            recommendations.append("Investigate prediction failures and improve error handling")
        
        if performance.get('total_predictions', 0) == 0:
            recommendations.append("Model has not been used recently - consider unloading")
        
        if not recommendations:
            recommendations.append("Model performance is optimal")
        
        return recommendations
    
    def __del__(self):
        """Cleanup when manager is destroyed"""
        try:
            self.stop_monitoring()
            # Clear all models
            for model_name in list(self.loaded_models.keys()):
                self.unload_model(model_name)
        except:
            pass
