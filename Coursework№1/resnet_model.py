import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
import os
from data_loader import EuroSATDataset, ImageDataset
from config import Config
from PIL import Image
import numpy as np

class ResNetClassifier:
    """Классификатор на основе ResNet"""
    
    def __init__(self):
        self.config = Config()
        
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)  # Исправлено!
        self.model.fc = nn.Linear(self.model.fc.in_features, self.config.NUM_CLASSES)
        self.device = self.config.DEVICE
        self.model.to(self.device)
        
        self.train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.is_trained = False
    
    def train(self):
        """Обучение модели"""
        print("Обучение ResNet...")
        
        dataset = EuroSATDataset()
        all_data = dataset.load_data()
        train_data, val_data = dataset.get_train_val_split(all_data)
        
        train_dataset = ImageDataset(train_data, self.train_transform)
        val_dataset = ImageDataset(val_data, self.val_transform)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        criterion = nn.CrossEntropyLoss()
        
        optimizer = optim.Adam([
            {'params': self.model.conv1.parameters(), 'lr': 0.0001},
            {'params': self.model.bn1.parameters(), 'lr': 0.0001},
            {'params': self.model.layer1.parameters(), 'lr': 0.0001},
            {'params': self.model.layer2.parameters(), 'lr': 0.0001},
            {'params': self.model.layer3.parameters(), 'lr': 0.0001},
            {'params': self.model.layer4.parameters(), 'lr': 0.0001},
            {'params': self.model.fc.parameters(), 'lr': 0.001}
        ], weight_decay=1e-4)
        
        best_val_acc = 0
        
        print("Начинаем обучение ResNet...")
        for epoch in range(3):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            train_acc = 100 * correct / total
            train_loss = running_loss / len(train_loader)
            
            val_acc, val_loss = self._validate(val_loader, criterion)
            
            print(f"Эпоха {epoch+1}/3:")
            print(f"  Train - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")
            print(f"  Val   - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model()
                print(f"  🎯 Новый лучший результат!")
        
        print(f"🏆 Лучшая точность ResNet на валидации: {best_val_acc:.2f}%")
        self.is_trained = True
        return best_val_acc / 100
    
    def _validate(self, val_loader, criterion):
        """Валидация модели"""
        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        val_loss = val_loss / len(val_loader)
        return val_acc, val_loss
    
    def predict(self, image_array):
        """Предсказание для одного изображения"""
        if not self.is_trained:
            self.load_model()
        
        self.model.eval()
        
        try:
            image = Image.fromarray(image_array)
            tensor = self.val_transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            
            return predicted.item(), probabilities[0].cpu().numpy()
        except Exception as e:
            print(f"Ошибка предсказания ResNet: {e}")
            return 0, np.zeros(self.config.NUM_CLASSES)
    
    def save_model(self):
        """Сохранение модели"""
        os.makedirs(self.config.MODELS_PATH, exist_ok=True)
        torch.save(self.model.state_dict(), 
                  os.path.join(self.config.MODELS_PATH, 'resnet.pth'))
        print("✅ ResNet model saved")
    
    def load_model(self):
        """Загрузка модели"""
        model_path = os.path.join(self.config.MODELS_PATH, 'resnet.pth')
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.is_trained = True
            print("✅ ResNet model loaded")
        else:
            raise FileNotFoundError("❌ ResNet model not found. Train first.")