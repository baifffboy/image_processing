import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from config import Config

class DataAnalyzer:
    def __init__(self):
        self.config = Config()
    
    def analyze(self):
        """Анализ датасета"""
        print("=== АНАЛИЗ ДАТАСЕТА EuroSAT ===")
        
        class_counts = {}
        for class_name in self.config.CLASSES:
            class_path = os.path.join(self.config.DATA_PATH, class_name)
            if os.path.exists(class_path):
                images = [f for f in os.listdir(class_path) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif'))]
                class_counts[class_name] = len(images)
        
        self._plot_class_distribution(class_counts)
        return class_counts
    
    def _plot_class_distribution(self, class_counts):
        """Визуализация распределения классов"""
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        bars = plt.bar(class_counts.keys(), class_counts.values())
        plt.title('Распределение изображений по классам')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Количество')
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        plt.subplot(1, 2, 2)
        plt.pie(class_counts.values(), labels=class_counts.keys(), autopct='%1.1f%%')
        plt.title('Процентное распределение')
        
        plt.tight_layout()
        plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    analyzer = DataAnalyzer()
    analyzer.analyze()