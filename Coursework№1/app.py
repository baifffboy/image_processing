from flask import Flask, request, jsonify
import os
from PIL import Image
import numpy as np
import io
import base64
from traditional_model import TraditionalClassifier
from simple_cnn import SimpleCNNClassifier
from resnet_model import ResNetClassifier
from config import Config

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

traditional = TraditionalClassifier()
simple_cnn = SimpleCNNClassifier()
resnet = ResNetClassifier()

def init_models():
    print("Загрузка моделей...")
    try:
        traditional.load_model()
        print("Traditional model loaded")
    except:
        print("Traditional model not available")
    
    try:
        simple_cnn.load_model()
        print("Simple CNN model loaded")
    except:
        print("Simple CNN model not available")
    
    try:
        resnet.load_model()
        print("ResNet model loaded")
    except:
        print("ResNet model not available")

HTML = '''
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EuroSAT Classifier</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #2c3e50;
            text-align: center;
            margin-bottom: 30px;
        }
        .upload-section {
            border: 2px dashed #ddd;
            padding: 40px;
            text-align: center;
            margin-bottom: 30px;
            border-radius: 10px;
            background-color: #fafafa;
        }
        .upload-section.dragover {
            border-color: #4CAF50;
            background-color: #f0fff0;
        }
        #imagePreview {
            max-width: 300px;
            max-height: 300px;
            margin: 20px auto;
            display: none;
        }
        .results {
            display: none;
            margin-top: 30px;
        }
        .model-result {
            background: #f8f9fa;
            padding: 20px;
            margin: 15px 0;
            border-radius: 8px;
            border-left: 4px solid #007bff;
        }
        .model-result.winner {
            border-left-color: #28a745;
            background: #f0fff0;
        }
        .model-result h3 {
            margin-top: 0;
            color: #2c3e50;
        }
        .confidence {
            font-size: 18px;
            font-weight: bold;
            color: #28a745;
        }
        .class-name {
            font-size: 20px;
            font-weight: bold;
            color: #2c3e50;
            margin: 10px 0;
        }
        button {
            background: #4CAF50;
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            margin: 10px;
        }
        button:hover {
            background: #45a049;
        }
        .loading {
            display: none;
            text-align: center;
            margin: 20px 0;
        }
        .winner-badge {
            background: #28a745;
            color: white;
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 12px;
            margin-left: 10px;
        }
        .no-results {
            text-align: center;
            color: #6c757d;
            padding: 40px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🌍 Классификатор земной поверхности EuroSAT</h1>
        
        <div class="upload-section" id="uploadSection">
            <h3>Загрузите изображение для классификации</h3>
            <input type="file" id="imageInput" accept="image/*" style="display: none;">
            <button onclick="document.getElementById('imageInput').click()">Выбрать файл</button>
            <p>или перетащите изображение сюда</p>
            <img id="imagePreview" alt="Preview">
        </div>

        <div class="loading" id="loading">
            <p>⏳ Анализируем изображение...</p>
        </div>

        <div class="results" id="results">
            <h2>🎯 Результаты классификации:</h2>
            <div id="resultsContainer"></div>
        </div>
    </div>

    <script>
        const uploadSection = document.getElementById('uploadSection');
        const imageInput = document.getElementById('imageInput');
        const imagePreview = document.getElementById('imagePreview');
        const results = document.getElementById('results');
        const resultsContainer = document.getElementById('resultsContainer');
        const loading = document.getElementById('loading');

        // Обработчик выбора файла
        imageInput.addEventListener('change', function(e) {
            if (e.target.files.length > 0) {
                handleImage(e.target.files[0]);
            }
        });

        // Drag and drop
        uploadSection.addEventListener('dragover', function(e) {
            e.preventDefault();
            uploadSection.classList.add('dragover');
        });

        uploadSection.addEventListener('dragleave', function() {
            uploadSection.classList.remove('dragover');
        });

        uploadSection.addEventListener('drop', function(e) {
            e.preventDefault();
            uploadSection.classList.remove('dragover');
            if (e.dataTransfer.files.length > 0) {
                handleImage(e.dataTransfer.files[0]);
            }
        });

        function handleImage(file) {
            if (!file.type.match('image.*')) {
                alert('Пожалуйста, выберите файл изображения');
                return;
            }

            const reader = new FileReader();
            reader.onload = function(e) {
                imagePreview.src = e.target.result;
                imagePreview.style.display = 'block';
                classifyImage(file);
            };
            reader.readAsDataURL(file);
        }

        function classifyImage(file) {
            loading.style.display = 'block';
            results.style.display = 'none';

            const formData = new FormData();
            formData.append('image', file);

            fetch('/predict', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                loading.style.display = 'none';
                
                if (data.error) {
                    alert('Ошибка: ' + data.error);
                    return;
                }

                displayResults(data);
            })
            .catch(error => {
                loading.style.display = 'none';
                alert('Ошибка при классификации: ' + error);
            });
        }

        function displayResults(data) {
            resultsContainer.innerHTML = '';
            
            if (data.image) {
                imagePreview.src = data.image;
            }

            const models = {
                'traditional': 'Traditional (SIFT + SVM)',
                'simple_cnn': 'Simple CNN', 
                'resnet': 'ResNet'
            };

            let bestModel = null;
            let bestConfidence = 0;

            // Находим модель с наибольшей уверенностью
            for (const [modelKey, modelName] of Object.entries(models)) {
                if (data.results[modelKey]) {
                    const confidence = data.results[modelKey].confidence;
                    if (confidence > bestConfidence) {
                        bestConfidence = confidence;
                        bestModel = modelKey;
                    }
                }
            }

            // Отображаем результаты
            let hasResults = false;
            for (const [modelKey, modelName] of Object.entries(models)) {
                if (data.results[modelKey]) {
                    hasResults = true;
                    const result = data.results[modelKey];
                    const isWinner = modelKey === bestModel;
                    
                    const resultDiv = document.createElement('div');
                    resultDiv.className = `model-result ${isWinner ? 'winner' : ''}`;
                    
                    resultDiv.innerHTML = `
                        <h3>
                            ${modelName}
                            ${isWinner ? '<span class="winner-badge">🏆 ЛУЧШИЙ</span>' : ''}
                        </h3>
                        <div class="class-name">${result.class}</div>
                        <div class="confidence">Уверенность: ${(result.confidence * 100).toFixed(2)}%</div>
                    `;
                    
                    resultsContainer.appendChild(resultDiv);
                }
            }

            if (!hasResults) {
                resultsContainer.innerHTML = '<div class="no-results">❌ Не удалось выполнить классификацию</div>';
            }

            results.style.display = 'block';
        }
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return HTML

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'})
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'})
    
    try:
        image = Image.open(file.stream).convert('RGB')
        image = image.resize(Config.IMG_SIZE)
        image_array = np.array(image)
        
        results = {}
        
        if hasattr(traditional, 'is_trained') and traditional.is_trained:
            try:
                trad_pred, trad_probs = traditional.predict(image_array)
                trad_confidence = float(np.max(trad_probs))
                results['traditional'] = {
                    'class': Config.CLASSES[trad_pred],
                    'confidence': trad_confidence
                }
            except Exception as e:
                print(f"Traditional prediction error: {e}")
        
        try:
            cnn_pred, cnn_probs = simple_cnn.predict(image_array)
            cnn_confidence = float(np.max(cnn_probs))
            results['simple_cnn'] = {
                'class': Config.CLASSES[cnn_pred],
                'confidence': cnn_confidence
            }
        except Exception as e:
            print(f"Simple CNN prediction error: {e}")
        
        try:
            resnet_pred, resnet_probs = resnet.predict(image_array)
            resnet_confidence = float(np.max(resnet_probs))
            results['resnet'] = {
                'class': Config.CLASSES[resnet_pred],
                'confidence': resnet_confidence
            }
        except Exception as e:
            print(f"ResNet prediction error: {e}")

        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return jsonify({
            'success': True,
            'image': f"data:image/jpeg;base64,{img_str}",
            'results': results
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    init_models()
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5050)