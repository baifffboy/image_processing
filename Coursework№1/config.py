import torch

class Config:
    DATA_PATH = "EuroSAT"
    MODELS_PATH = "trained_models"
    
    NUM_EPOCHS = 3
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    IMG_SIZE = (64, 64)
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    CLASSES = [
        'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 
        'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 
        'River', 'SeaLake'
    ]
    NUM_CLASSES = len(CLASSES)