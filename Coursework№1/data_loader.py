import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch
from config import Config

class EuroSATDataset:
    """Загрузка данных EuroSAT с ПРАВИЛЬНЫМ разделением"""
    
    def __init__(self):
        self.config = Config()
        self.dataset = []
    
    def load_data(self, mode='all'):
        """Загрузка всех данных"""
        print("Загрузка данных...")
        
        all_data = []
        for class_idx, class_name in enumerate(self.config.CLASSES):
            class_path = os.path.join(self.config.DATA_PATH, class_name)
            
            if not os.path.exists(class_path):
                print(f"Пропускаем {class_name} - папка не найдена")
                continue
                
            for filename in os.listdir(class_path):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.tif')):
                    filepath = os.path.join(class_path, filename)
                    try:
                        with Image.open(filepath) as img:
                            img = img.convert('RGB')
                            img = img.resize(self.config.IMG_SIZE)
                            img_array = np.array(img)
                            
                            all_data.append({
                                'image': img_array,
                                'class_idx': class_idx,
                                'class_name': class_name,
                                'filepath': filepath
                            })
                    except Exception as e:
                        print(f"Ошибка загрузки {filepath}: {e}")
        
        print(f"Загружено {len(all_data)} изображений")
        
        indices = torch.randperm(len(all_data)).tolist()
        self.dataset = [all_data[i] for i in indices]
        
        return self.dataset
    
    def get_train_val_split(self, data, train_ratio=0.8):
        """Разделение на train/val"""
        split_idx = int(train_ratio * len(data))
        train_data = data[:split_idx]
        val_data = data[split_idx:]
        
        print(f"Обучающая выборка: {len(train_data)}")
        print(f"Валидационная выборка: {len(val_data)}")
        
        return train_data, val_data

class ImageDataset(Dataset):
    """PyTorch Dataset"""
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_data = self.data[idx]
        img_array = img_data['image']
        label = img_data['class_idx']
        
        img = Image.fromarray(img_array)
        
        if self.transform:
            img = self.transform(img)
            
        return img, label