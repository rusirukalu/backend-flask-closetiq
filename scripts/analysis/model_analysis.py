# model_analysis.py - Comprehensive Model Performance Analysis

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
import os

class ModelAnalyzer:
    def __init__(self, model_path, test_data_path):
        self.model_path = model_path
        self.test_data_path = test_data_path
        self.model = None
        self.class_names = ['bottom', 'dress', 'formal', 'outerwear', 'shoes', 'top']
        
    def load_model_and_data(self):
        """Load the trained model and test data"""
        print("🔄 Loading model and test data...")
        
        # Load model
        try:
            self.model = load_model(self.model_path)
            print(f"✅ Model loaded from {self.model_path}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
            
        # Load test data
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        self.test_generator = test_datagen.flow_from_directory(
            self.test_data_path,
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical',
            shuffle=False,
            classes=self.class_names
        )
        
        print(f"✅ Test data loaded: {self.test_generator.samples} images")
        return True
    
    def analyze_performance(self):
        """Comprehensive performance analysis"""
        print("\n🔍 Starting comprehensive performance analysis...")
        
        # Get predictions
        print("Making predictions on test set...")
        predictions = self.model.predict(self.test_generator)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = self.test_generator.classes
        
        # Classification Report
        print("\n📊 Classification Report:")
        class_report = classification_report(
            true_classes, predicted_classes, 
            target_names=self.class_names,
            output_dict=True
        )
        
        print(classification_report(true_classes, predicted_classes, target_names=self.class_names))
        
        # Confusion Matrix
        cm = confusion_matrix(true_classes, predicted_classes)
        
        # Detailed Analysis
        analysis_results = {
            'timestamp': datetime.now().isoformat(),
            'model_path': self.model_path,
            'total_test_samples': len(true_classes),
            'overall_accuracy': float(np.mean(predicted_classes == true_classes)),
            'classification_report': class_report,
            'confusion_matrix': cm.tolist(),
            'class_names': self.class_names,
            'confusion_analysis': self._analyze_confusions(cm),
            'per_class_analysis': self._per_class_analysis(class_report),
            'problem_categories': self._identify_problem_categories(class_report, cm)
        }
        
        return analysis_results
    
    def _analyze_confusions(self, cm):
        """Analyze specific confusion patterns"""
        confusions = []
        
        for i in range(len(self.class_names)):
            for j in range(len(self.class_names)):
                if i != j and cm[i][j] > 0:
                    confusion_rate = cm[i][j] / np.sum(cm[i]) * 100
                    confusions.append({
                        'true_class': self.class_names[i],
                        'predicted_class': self.class_names[j],
                        'count': int(cm[i][j]),
                        'confusion_rate': float(confusion_rate)
                    })
        
        # Sort by confusion rate
        confusions.sort(key=lambda x: x['confusion_rate'], reverse=True)
        return confusions[:10]  # Top 10 confusions
    
    def _per_class_analysis(self, class_report):
        """Per-class performance analysis"""
        per_class = {}
        
        for class_name in self.class_names:
            if class_name in class_report:
                metrics = class_report[class_name]
                per_class[class_name] = {
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1_score': metrics['f1-score'],
                    'support': metrics['support'],
                    'performance_grade': self._grade_performance(metrics['f1-score'])
                }
        
        return per_class
    
    def _grade_performance(self, f1_score):
        """Grade performance based on F1-score"""
        if f1_score >= 0.9:
            return 'Excellent'
        elif f1_score >= 0.8:
            return 'Good'
        elif f1_score >= 0.7:
            return 'Fair'
        elif f1_score >= 0.6:
            return 'Poor'
        else:
            return 'Very Poor'
    
    def _identify_problem_categories(self, class_report, cm):
        """Identify categories that need the most improvement"""
        problems = []
        
        for i, class_name in enumerate(self.class_names):
            if class_name in class_report:
                metrics = class_report[class_name]
                f1_score = metrics['f1-score']
                
                if f1_score < 0.7:  # Poor performance threshold
                    # Find what this class is most confused with
                    row = cm[i]
                    most_confused_idx = np.argsort(row)[-2]  # Second highest (first is correct class)
                    most_confused_class = self.class_names[most_confused_idx]
                    confusion_count = row[most_confused_idx]
                    
                    problems.append({
                        'class': class_name,
                        'f1_score': f1_score,
                        'most_confused_with': most_confused_class,
                        'confusion_count': int(confusion_count),
                        'priority': 'High' if f1_score < 0.5 else 'Medium'
                    })
        
        return problems
    
    def generate_visualizations(self, analysis_results):
        """Generate comprehensive visualizations"""
        print("\n📈 Generating visualizations...")
        
        # 1. Confusion Matrix Heatmap
        plt.figure(figsize=(12, 10))
        
        plt.subplot(2, 2, 1)
        cm = np.array(analysis_results['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        # 2. Per-Class F1 Scores
        plt.subplot(2, 2, 2)
        per_class = analysis_results['per_class_analysis']
        classes = list(per_class.keys())
        f1_scores = [per_class[cls]['f1_score'] for cls in classes]
        
        bars = plt.bar(classes, f1_scores, color=['red' if f1 < 0.7 else 'orange' if f1 < 0.8 else 'green' for f1 in f1_scores])
        plt.title('F1-Score by Category')
        plt.ylabel('F1-Score')
        plt.xticks(rotation=45)
        plt.axhline(y=0.7, color='red', linestyle='--', alpha=0.7, label='Poor threshold')
        plt.axhline(y=0.8, color='orange', linestyle='--', alpha=0.7, label='Good threshold')
        plt.legend()
        
        # 3. Top Confusions
        plt.subplot(2, 2, 3)
        confusions = analysis_results['confusion_analysis'][:5]  # Top 5
        confusion_labels = [f"{c['true_class']} → {c['predicted_class']}" for c in confusions]
        confusion_rates = [c['confusion_rate'] for c in confusions]
        
        plt.barh(confusion_labels, confusion_rates, color='coral')
        plt.title('Top 5 Category Confusions')
        plt.xlabel('Confusion Rate (%)')
        
        # 4. Overall Performance Summary
        plt.subplot(2, 2, 4)
        metrics = ['Accuracy', 'Avg Precision', 'Avg Recall', 'Avg F1']
        values = [
            analysis_results['overall_accuracy'],
            analysis_results['classification_report']['macro avg']['precision'],
            analysis_results['classification_report']['macro avg']['recall'],
            analysis_results['classification_report']['macro avg']['f1-score']
        ]
        
        bars = plt.bar(metrics, values, color=['lightblue', 'lightgreen', 'lightcoral', 'lightyellow'])
        plt.title('Overall Performance Metrics')
        plt.ylabel('Score')
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('model_performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self, analysis_results):
        """Generate comprehensive text report"""
        print("\n📝 Generating Performance Report...")
        
        report = f"""
# MODEL PERFORMANCE ANALYSIS REPORT
Generated: {analysis_results['timestamp']}
Model: {analysis_results['model_path']}

## OVERALL PERFORMANCE
- Total Test Samples: {analysis_results['total_test_samples']}
- Overall Accuracy: {analysis_results['overall_accuracy']:.3f} ({analysis_results['overall_accuracy']*100:.1f}%)
- Average Precision: {analysis_results['classification_report']['macro avg']['precision']:.3f}
- Average Recall: {analysis_results['classification_report']['macro avg']['recall']:.3f}
- Average F1-Score: {analysis_results['classification_report']['macro avg']['f1-score']:.3f}

## PER-CLASS PERFORMANCE
"""
        
        for class_name, metrics in analysis_results['per_class_analysis'].items():
            report += f"""
### {class_name.upper()}
- Precision: {metrics['precision']:.3f}
- Recall: {metrics['recall']:.3f}
- F1-Score: {metrics['f1_score']:.3f}
- Support: {metrics['support']} samples
- Grade: {metrics['performance_grade']}
"""
        
        report += f"""
## TOP CONFUSION PATTERNS
"""
        
        for confusion in analysis_results['confusion_analysis'][:5]:
            report += f"""
- {confusion['true_class']} → {confusion['predicted_class']}: {confusion['count']} cases ({confusion['confusion_rate']:.1f}%)
"""
        
        report += f"""
## PROBLEM CATEGORIES (Need Improvement)
"""
        
        if analysis_results['problem_categories']:
            for problem in analysis_results['problem_categories']:
                report += f"""
### {problem['class'].upper()} - Priority: {problem['priority']}
- F1-Score: {problem['f1_score']:.3f}
- Most confused with: {problem['most_confused_with']}
- Confusion cases: {problem['confusion_count']}
"""
        else:
            report += "✅ No major problem categories identified!"
        
        report += f"""
## RECOMMENDATIONS FOR IMPROVEMENT

1. **Data Collection Priority:**
"""
        
        for problem in analysis_results['problem_categories']:
            report += f"   - Collect more {problem['class']} images (especially vs {problem['most_confused_with']})\n"
        
        report += f"""
2. **Data Augmentation Focus:**
   - Enhance confused category pairs with targeted augmentation
   - Focus on distinguishing features between similar categories

3. **Model Architecture:**
   - Consider attention mechanisms for better feature focus
   - Implement class-weighted training for imbalanced categories

4. **Training Strategy:**
   - Use confusion-focused loss functions
   - Implement hard negative mining for difficult cases
"""
        
        # Save report
        with open('model_analysis_report.txt', 'w') as f:
            f.write(report)
        
        print(report)
        return report

# Run the analysis
def run_step1_analysis(model_path="final_working_model.h5", test_data_path="quick_fashion_11cat/test"):
    """Run Step 1: Complete model performance analysis"""
    
    print("🚀 STEP 1: MODEL PERFORMANCE ANALYSIS")
    print("=" * 50)
    
    analyzer = ModelAnalyzer(model_path, test_data_path)
    
    # Load model and data
    if not analyzer.load_model_and_data():
        return None
    
    # Analyze performance
    analysis_results = analyzer.analyze_performance()
    
    # Generate visualizations
    analyzer.generate_visualizations(analysis_results)
    
    # Generate report
    analyzer.generate_report(analysis_results)
    
    # Save results
    with open('model_analysis_results.json', 'w') as f:
        json.dump(analysis_results, f, indent=2, default=str)
    
    print("\n✅ Step 1 Analysis Complete!")
    print("📁 Files generated:")
    print("   - model_analysis_results.json")
    print("   - model_analysis_report.txt") 
    print("   - model_performance_analysis.png")
    
    return analysis_results

# Execute Step 1
if __name__ == "__main__":
    results = run_step1_analysis()
