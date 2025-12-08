import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os
from data_loader import EuroSATDataset
from config import Config

class TraditionalClassifier:
    """Традиционный метод (SIFT + SVM)"""
    
    def __init__(self):
        self.config = Config()
        self.sift = cv2.SIFT_create()
        self.svm = SVC(kernel='rbf', probability=True, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def extract_features(self, images):
        """Извлечение SIFT-признаков"""
        features = []
        for img_array in images:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            keypoints, descriptors = self.sift.detectAndCompute(gray, None)
            
            if descriptors is not None:
                avg_descriptor = np.mean(descriptors, axis=0)
                features.append(avg_descriptor)
            else:
                features.append(np.zeros(128))
        
        return np.array(features)
    
    def train(self):
        """Обучение модели"""
        print("Обучение Traditional Classifier...")
        
        dataset = EuroSATDataset()
        data = dataset.load_data()
        
        X = [item['image'] for item in data]
        y = [item['class_idx'] for item in data]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print("Извлечение SIFT-признаков...")
        X_train_features = self.extract_features(X_train)
        X_test_features = self.extract_features(X_test)
        
        X_train_scaled = self.scaler.fit_transform(X_train_features)
        X_test_scaled = self.scaler.transform(X_test_features)
        
        print("Обучение SVM...")
        self.svm.fit(X_train_scaled, y_train)
        

        train_score = self.svm.score(X_train_scaled, y_train)
        test_score = self.svm.score(X_test_scaled, y_test)
        
        print(f"Точность Traditional Classifier:")
        print(f"  Обучающая выборка: {train_score:.4f}")
        print(f"  Тестовая выборка: {test_score:.4f}")
        
        self.is_trained = True
        return test_score
    
    def predict(self, image_array):
        """Предсказание для одного изображения"""
        if not self.is_trained:
            self.load_model()
        
        features = self.extract_features([image_array])
        features_scaled = self.scaler.transform(features)
        
        prediction = self.svm.predict(features_scaled)[0]
        probability = self.svm.predict_proba(features_scaled)[0]
        
        return prediction, probability
    
    def save_model(self):
        """Сохранение модели"""
        os.makedirs(self.config.MODELS_PATH, exist_ok=True)
        joblib.dump({
            'svm': self.svm,
            'scaler': self.scaler
        }, os.path.join(self.config.MODELS_PATH, 'traditional_model.pkl'))
        print("Traditional model saved")
    
    def load_model(self):
        """Загрузка модели"""
        model_path = os.path.join(self.config.MODELS_PATH, 'traditional_model.pkl')
        if os.path.exists(model_path):
            model_data = joblib.load(model_path)
            self.svm = model_data['svm']
            self.scaler = model_data['scaler']
            self.is_trained = True
            print("Traditional model loaded")
        else:
            raise FileNotFoundError("Model not found. Train first.")