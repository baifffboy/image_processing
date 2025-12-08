import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from traditional_model import TraditionalClassifier
from simple_cnn import SimpleCNNClassifier
from resnet_model import ResNetClassifier
from data_loader import EuroSATDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import os
from config import Config

class ModelComparator:
    """Сравнение моделей"""
    
    def __init__(self):
        self.config = Config()
    
    def compare_models(self):
        """Сравнение всех моделей"""
        print("=== СРАВНЕНИЕ МОДЕЛЕЙ ===")
        
        dataset = EuroSATDataset()
        data = dataset.load_data()
        
        X = [item['image'] for item in data]
        y = [item['class_idx'] for item in data]
        
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        results = {}
        
        print("\n1. Traditional Classifier...")
        traditional = TraditionalClassifier()
        try:
            traditional.load_model()
            trad_preds = []
            for img in X_test:
                pred, _ = traditional.predict(img)
                trad_preds.append(pred)
            
            trad_acc = accuracy_score(y_test, trad_preds)
            trad_f1 = f1_score(y_test, trad_preds, average='weighted')
            results['Traditional'] = {'accuracy': trad_acc, 'f1_score': trad_f1}
        except Exception as e:
            print(f"Traditional classifier error: {e}")
            results['Traditional'] = {'accuracy': 0, 'f1_score': 0}
        
        print("\n2. Simple CNN...")
        simple_cnn = SimpleCNNClassifier()
        try:
            simple_cnn.load_model()
            cnn_preds = []
            for img in X_test:
                pred, _ = simple_cnn.predict(img)
                cnn_preds.append(pred)
            
            cnn_acc = accuracy_score(y_test, cnn_preds)
            cnn_f1 = f1_score(y_test, cnn_preds, average='weighted')
            results['Simple CNN'] = {'accuracy': cnn_acc, 'f1_score': cnn_f1}
        except Exception as e:
            print(f"Simple CNN error: {e}")
            results['Simple CNN'] = {'accuracy': 0, 'f1_score': 0}
        
        print("\n3. ResNet...")
        resnet = ResNetClassifier()
        try:
            resnet.load_model()
            resnet_preds = []
            for img in X_test:
                pred, _ = resnet.predict(img)
                resnet_preds.append(pred)
            
            resnet_acc = accuracy_score(y_test, resnet_preds)
            resnet_f1 = f1_score(y_test, resnet_preds, average='weighted')
            results['ResNet'] = {'accuracy': resnet_acc, 'f1_score': resnet_f1}
        except Exception as e:
            print(f"ResNet error: {e}")
            results['ResNet'] = {'accuracy': 0, 'f1_score': 0}
        
        self._plot_results(results)
        
        best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
        
        print("\n" + "="*50)
        print("РЕЗУЛЬТАТЫ СРАВНЕНИЯ:")
        print("="*50)
        for model_name, metrics in results.items():
            print(f"{model_name}:")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  F1-Score: {metrics['f1_score']:.4f}")
        
        print(f"\n🏆 ЛУЧШАЯ МОДЕЛЬ: {best_model[0]}")
        print(f"   Accuracy: {best_model[1]['accuracy']:.4f}")
        print(f"   F1-Score: {best_model[1]['f1_score']:.4f}")
        
        return results, best_model[0]
    
    def _plot_results(self, results):
        """Визуализация результатов"""
        models = list(results.keys())
        accuracies = [results[m]['accuracy'] for m in models]
        f1_scores = [results[m]['f1_score'] for m in models]
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        bars = plt.bar(models, accuracies, color=['#FF9999', '#66B2FF', '#99FF99'])
        plt.title('Сравнение точности моделей', fontsize=14, fontweight='bold')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.subplot(1, 2, 2)
        bars = plt.bar(models, f1_scores, color=['#FF9999', '#66B2FF', '#99FF99'])
        plt.title('Сравнение F1-score моделей', fontsize=14, fontweight='bold')
        plt.ylabel('F1-Score')
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    comparator = ModelComparator()
    results, best_model = comparator.compare_models()