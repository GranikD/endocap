import os
import time
import threading
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, request, render_template_string
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

MODEL_PATH = r'kvasir_resnet50_final.pth' 
BLUETOOTH_FOLDER = r'C:\Bluetooth_Photos' 

class_names = [
    'barretts', 'barretts-short-segment', 'bbps-0-1', 'bbps-2-3', 'cecum',
    'dyed-lifted-polyps', 'dyed-resection-margins', 'esophagitis-a',
    'esophagitis-b-d', 'hemorrhoids', 'ileum', 'impacted-stool', 'polyps',
    'pylorus', 'retroflex-rectum', 'retroflex-stomach', 'ulcerative-colitis-grade-0-1',
    'ulcerative-colitis-grade-1', 'ulcerative-colitis-grade-1-2', 'ulcerative-colitis-grade-2',
    'ulcerative-colitis-grade-2-3', 'ulcerative-colitis-grade-3', 'z-line'
]

translate_dict = {
    'barretts': 'Пищевод Барретта',
    'barretts-short-segment': 'Пищевод Барретта (короткий сегмент)',
    'bbps-0-1': 'Низкое качество очистки (BBPS 0-1)',
    'bbps-2-3': 'Высокое качество очистки (BBPS 2-3)',
    'cecum': 'Слепая кишка (норма)',
    'dyed-lifted-polyps': 'Полип (маркировка красителем)',
    'dyed-resection-margins': 'Зона резекции (после удаления)',
    'esophagitis-a': 'Эзофагит (стадия А)',
    'esophagitis-b-d': 'Эзофагит (стадии B-D)',
    'hemorrhoids': 'Геморрой',
    'ileum': 'Подвздошная кишка (норма)',
    'impacted-stool': 'Загрязнение каловыми массами',
    'polyps': 'Полип',
    'pylorus': 'Привратник желудка (норма)',
    'retroflex-rectum': 'Прямая кишка (ретрофлексия)',
    'retroflex-stomach': 'Желудок (ретрофлексия)',
    'ulcerative-colitis-grade-0-1': 'Язвенный колит (стадия 0-1)',
    'ulcerative-colitis-grade-1': 'Язвенный колит (стадия 1)',
    'ulcerative-colitis-grade-1-2': 'Язвенный колит (стадия 1-2)',
    'ulcerative-colitis-grade-2': 'Язвенный колит (стадия 2)',
    'ulcerative-colitis-grade-2-3': 'Язвенный колит (стадия 2-3)',
    'ulcerative-colitis-grade-3': 'Язвенный колит (стадия 3)',
    'z-line': 'Z-линия (переход в желудок)'
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
app = Flask(__name__)
model = None

def load_model():
    global model
    model = models.resnet50()
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Файл {MODEL_PATH} не найден!")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()

def predict_image(image_obj):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    try:
        img = Image.open(image_obj).convert('RGB')
        img_tensor = preprocess(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.nn.functional.softmax(output[0], dim=0)
            conf, index = torch.max(probs, 0)
        class_en = class_names[index.item()]
        class_ru = translate_dict.get(class_en, class_en)
        return class_ru, conf.item() * 100
    except Exception as e:
        return f"Ошибка: {e}", 0

class BluetoothHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            time.sleep(1)
            result_ru, conf = predict_image(event.src_path)
            print(f"Bluetooth Result: {result_ru} ({conf:.2f}%)")

def start_bluetooth_watcher():
    if not os.path.exists(BLUETOOTH_FOLDER):
        os.makedirs(BLUETOOTH_FOLDER)
    observer = Observer()
    observer.schedule(BluetoothHandler(), BLUETOOTH_FOLDER, recursive=False)
    observer.start()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Нейросеть Эндоскопии</title>
    <style>
        body { font-family: Arial; margin: 40px; background: #f0f2f5; }
        .box { max-width: 500px; margin: auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
        .result { margin-top: 20px; padding: 15px; background: #e8f5e9; border-left: 5px solid #4caf50; }
    </style>
</head>
<body>
    <div class="box">
        <h2>Анализ снимка</h2>
        <form method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept="image/*" required>
            <button type="submit" style="margin-top: 10px; padding: 8px;">Анализировать</button>
        </form>
        {% if result %}
        <div class="result">
            <h3>Диагноз: {{ result }}</h3>
            <p>Уверенность сети: <b>{{ conf|round(2) }}%</b></p>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    result_ru, conf = None, None
    if request.method == 'POST':
        file = request.files['file']
        if file:
            result_ru, conf = predict_image(file.stream)
    return render_template_string(HTML_TEMPLATE, result=result_ru, conf=conf)

if __name__ == "__main__":
    load_model()
    bt_thread = threading.Thread(target=start_bluetooth_watcher, daemon=True)
    bt_thread.start()
    app.run(host='0.0.0.0', port=5000, debug=False)
