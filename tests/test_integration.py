# tests/test_integration.py - Comprehensive Integration Tests
import unittest
import json
import io
import os
import sys
from PIL import Image
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from config.settings import Config

class FashionAIIntegrationTests(unittest.TestCase):
    """Comprehensive integration tests for Fashion AI Backend"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.app = create_app()
        cls.client = cls.app.test_client()
        cls.app.config['TESTING'] = True
        
        # Create test image
        cls.test_image = cls.create_test_image()
    
    @classmethod
    def create_test_image(cls):
        """Create a test image for testing"""
        # Create a simple test image
        image = Image.new('RGB', (224, 224), color='red')
        
        # Save to bytes
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        return img_bytes
    
    def test_health_check(self):
        """Test health check endpoint"""
        response = self.client.get('/health')
        
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertIn('services', data)
        self.assertIn('uptime_seconds', data)
    
    def test_detailed_status(self):
        """Test detailed status endpoint"""
        response = self.client.get('/api/status')
        
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertIn('detailed_status', data)
    
    def test_image_classification(self):
        """Test image classification endpoint"""
        # Reset image pointer
        self.test_image.seek(0)
        
        response = self.client.post('/api/classify', 
                                  data={'file': (self.test_image, 'test.png')},
                                  content_type='multipart/form-data')
        
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertIn('classification', data)
        self.assertIn('image_quality', data)
        self.assertIn('processing_time_ms', data)
    
    def test_batch_classification(self):
        """Test batch classification endpoint"""
        # Create multiple test images
        images = []
        for i in range(3):
            img_bytes = io.BytesIO()
            image = Image.new('RGB', (224, 224), color=['red', 'green', 'blue'][i])
            image.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            images.append((img_bytes, f'test_{i}.png'))
        
        response = self.client.post('/api/classify/batch',
                                  data={'files': images},
                                  content_type='multipart/form-data')
        
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertIn('results', data)
        self.assertIn('summary', data)
        self.assertEqual(len(data['results']), 3)
    
    def test_compatibility_check(self):
        """Test style compatibility endpoint"""
        test_data = {
            'item1_id': 'test_item_1',
            'item2_id': 'test_item_2',
            'context': 'casual'
        }
        
        response = self.client.post('/api/compatibility/check',
                                  data=json.dumps(test_data),
                                  content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertIn('compatibility', data)
    
    def test_outfit_generation(self):
        """Test outfit generation endpoint"""
        test_data = {
            'user_id': 'test_user',
            'items': [
                {'category': 'shirts_blouses', 'attributes': {'colors': ['white']}},
                {'category': 'pants_jeans', 'attributes': {'colors': ['blue']}},
                {'category': 'shoes_formal', 'attributes': {'colors': ['black']}}
            ],
            'occasion': 'work',
            'season': 'spring',
            'count': 3
        }
        
        response = self.client.post('/api/outfits/generate',
                                  data=json.dumps(test_data),
                                  content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertIn('recommendations', data)
    
    def test_performance_report(self):
        """Test performance report endpoint"""
        response = self.client.get('/api/performance/report')
        
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertIn('performance_report', data)
    
    def test_model_optimization(self):
        """Test model optimization endpoint"""
        response = self.client.post('/api/models/optimize')
        
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertTrue(data['success'])
    
    def test_error_handling(self):
        """Test error handling"""
        # Test missing file
        response = self.client.post('/api/classify')
        self.assertEqual(response.status_code, 400)
        
        # Test invalid endpoint
        response = self.client.get('/api/invalid-endpoint')
        self.assertEqual(response.status_code, 404)
        
        # Test invalid method
        response = self.client.delete('/health')
        self.assertEqual(response.status_code, 405)
    
    def test_api_consistency(self):
        """Test API response consistency"""
        endpoints = [
            ('/health', 'GET'),
            ('/api/status', 'GET'),
            ('/api/performance/report', 'GET')
        ]
        
        for endpoint, method in endpoints:
            if method == 'GET':
                response = self.client.get(endpoint)
            else:
                response = self.client.post(endpoint)
            
            data = json.loads(response.data)
            
            # Check standard response format
            self.assertIn('success', data)
            self.assertIn('timestamp', data)
            
            if data['success']:
                self.assertEqual(response.status_code, 200)

if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
