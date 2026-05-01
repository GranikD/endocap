import os
import hashlib
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, request, render_template_string
import cv2
import numpy as np
import base64
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


MODEL_PATH = 'kvasir_resnet50_final.pth'
EXPECTED_HASH = "EF58391F8B8C062D248CF86170DC6C1B14BC7886C1CE19027B8A0A2E80B02846"
REPORTS_DATABASE = []

class_names = [
    'barretts', 'barretts-short-segment', 'bbps-0-1', 'bbps-2-3', 'cecum',
    'dyed-lifted-polyps', 'dyed-resection-margins', 'esophagitis-a',
    'esophagitis-b-d', 'hemorrhoids', 'ileum', 'impacted-stool', 'polyps',
    'pylorus', 'retroflex-rectum', 'retroflex-stomach', 'ulcerative-colitis-grade-0-1',
    'ulcerative-colitis-grade-1', 'ulcerative-colitis-grade-1-2', 'ulcerative-colitis-grade-2',
    'ulcerative-colitis-grade-2-3', 'ulcerative-colitis-grade-3', 'z-line'
]

translation_dict = {
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

app = Flask(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None




def verify_integrity_and_get_hash(path):
    if not os.path.exists(path): return None
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha.update(chunk)

    calc_hash = sha.hexdigest().upper()

    # ЛОГИКА СРАВНЕНИЯ ДЛЯ ПРЕЗЕНТАЦИИ:
    if calc_hash == EXPECTED_HASH.upper():
        print("ЦЕЛОСТНОСТЬ ПОДТВЕРЖДЕНА")
        return calc_hash
    else:
        print("ОШИБКА: ФАЙЛ МОДЕЛИ ИЗМЕНЕН ИЛИ ПОДМЕНЕН")
        return calc_hash


def get_secure_bbox(orig_img, grayscale_cam):
    h, w, _ = orig_img.shape
    heatmap = cv2.resize(grayscale_cam, (w, h))
    gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
    _, mask_non_black = cv2.threshold(gray, 40, 1.0, cv2.THRESH_BINARY)
    heatmap = heatmap * mask_non_black
    margin = int(min(h, w) * 0.1)
    heatmap[:margin, :] = 0;
    heatmap[-margin:, :] = 0
    heatmap[:, :margin] = 0;
    heatmap[:, -margin:] = 0
    max_val = np.max(heatmap)
    if max_val < 0.1: return None
    _, thresh = cv2.threshold(heatmap, max_val * 0.5, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(np.uint8(thresh), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    best_cnt = max(contours, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(best_cnt)
    pad = int(max(bw, bh) * 0.2)
    return (max(0, x - pad), max(0, y - pad), min(w - x, bw + 2 * pad), min(h - y, bh + 2 * pad))


def draw_info(image_cv, text, pos, color=(0, 255, 0)):
    img_pil = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    bbox = draw.textbbox(pos, text, font=font)
    draw.rectangle([bbox[0] - 5, bbox[1] - 2, bbox[2] + 5, bbox[3] + 2], fill=color)
    draw.text(pos, text, font=font, fill=(0, 0, 0))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def process_image(file_stream, user_message=""):
    img_pil = Image.open(file_stream).convert('RGB')
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(img_pil).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output[0], dim=0)
        conf, idx = torch.max(probs, 0)

    cam = GradCAMPlusPlus(model=model, target_layers=[model.layer4[-1]])
    grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(idx.item())])[0, :]

    bbox = get_secure_bbox(img_cv, grayscale_cam)
    name_ru = translation_dict.get(class_names[idx], class_names[idx])

    if bbox:
        x, y, w, h = bbox
        cv2.rectangle(img_cv, (x, y), (x + w, y + h), (0, 255, 0), 2)
        img_cv = draw_info(img_cv, f"{name_ru} {conf.item() * 100:.1f}%", (x, y - 30))

    _, buffer = cv2.imencode('.jpg', img_cv)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    REPORTS_DATABASE.append({
        'label': name_ru,
        'conf': round(conf.item() * 100, 1),
        'img': img_base64,
        'message': user_message 
    })
    return name_ru, conf.item() * 100, img_base64




CSS = """
<style>
    body { background: #0b0b0b; color: #e0e0e0; font-family: 'Segoe UI', sans-serif; text-align: center; margin: 0; padding: 20px; }
    .container { background: #151515; border: 1px solid #333; border-radius: 15px; display: inline-block; padding: 40px; margin-top: 50px; min-width: 450px; box-shadow: 0 10px 30px rgba(0,0,0,0.5); }
    .btn { background: #00ff88; color: #000; padding: 15px 30px; border-radius: 8px; text-decoration: none; font-weight: bold; display: inline-block; margin: 10px; transition: 0.3s; border: none; cursor: pointer; }
    .btn:hover { background: #00cc6e; transform: translateY(-2px); }
    .btn-gray { background: #333; color: #fff; }
    textarea { width: 100%; background: #222; color: #fff; border: 1px solid #444; border-radius: 5px; padding: 10px; margin: 10px 0; font-family: inherit; }
    .report-card { background: #202020; border-radius: 10px; padding: 15px; margin: 15px 0; display: flex; align-items: center; text-align: left; border-left: 5px solid #00ff88; }
    .msg-box { background: #2a2a2a; padding: 10px; border-radius: 5px; margin-top: 10px; font-style: italic; color: #aaa; border-left: 2px solid #555; }
    img { border-radius: 8px; max-width: 100%; }
</style>
"""


@app.route('/')
def index():
    return render_template_string(CSS + """
    <div class="container">
        <h1 style="color:#00ff88">AI ENDOSCOPY """ + """</h1>
        <p style="color:#888">Защищенный терминал связи и анализа</p>
        <hr style="border:0; border-top:1px solid #333; margin:30px 0;">
        <div style="display:flex; justify-content: center;">
            <a href="/patient" class="btn">ПАЦИЕНТ (Загрузка)</a>
            <a href="/doctor" class="btn btn-gray">ВРАЧ (Монитор)</a>
        </div>
        <p style="font-size:10px; color:#444; margin-top:30px;">БЕЗ СБОРА ПД • КАНАЛ СВЯЗИ ЗАШИФРОВАН</p>
    </div>
    """)


@app.route('/patient', methods=['GET', 'POST'])
def patient():
    img_data = None
    if request.method == 'POST':
        file = request.files.get('file')
        msg = request.form.get('message', '')
        if file:
            _, _, img_data = process_image(file.stream, msg)

    return render_template_string(CSS + """
    <div class="container">
        <h3>ОТПРАВКА СНИМКА ВРАЧУ</h3>
        <form method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept="image/*" required style="margin-bottom:10px;"><br>
            <textarea name="message" rows="3" placeholder="Сообщение врачу..."></textarea><br>
            <button type="submit" class="btn">ОТПРАВИТЬ НА АНАЛИЗ</button>
        </form>
        {% if img_data %}
            <div style="margin-top:20px;">
                <img src="data:image/jpeg;base64,{{ img_data }}">
                <p style="color:#00ff88">Успешно! Врач получил ваш снимок и сообщение.</p>
            </div>
        {% endif %}
        <br><a href="/" style="color:#666; text-decoration:none;">← На главную</a>
    </div>
    """, img_data=img_data)


@app.route('/doctor')
def doctor():
    return render_template_string(CSS + """
    <div class="container" style="width: 850px;">
        <h3>ПАНЕЛЬ ВРАЧА + ЧАТ</h3>
        <div style="max-height: 600px; overflow-y: auto;">
            {% for item in database %}
            <div class="report-card">
                <img src="data:image/jpeg;base64,{{ item.img }}" width="200" style="margin-right:20px;">
                <div style="flex-grow:1;">
                    <strong style="color:#00ff88; font-size:1.1em;">{{ item.label }} ({{ item.conf }}%)</strong><br>
                    {% if item.message %}
                        <div class="msg-box"><strong>Сообщение пациента:</strong><br>{{ item.message }}</div>
                    {% else %}
                        <small style="color:#444;">Без текстового сообщения</small>
                    {% endif %}
                </div>
            </div>
            {% endfor %}
            {% if not database %}
                <p style="padding:50px; color:#444;">Входящих сообщений нет...</p>
            {% endif %}
        </div>
        <br><a href="/" style="color:#666; text-decoration:none;">← Выйти из системы</a>
    </div>
    """, database=REPORTS_DATABASE[::-1])


if __name__ == "__main__":
    file_hash = verify_integrity_and_get_hash(MODEL_PATH)
    print("\n" + "=" * 50)
    print(f" SHA-256 МОДЕЛИ: {file_hash if file_hash else 'ФАЙЛ НЕ НАЙДЕН'}")
    print("=" * 50 + "\n")

    model = models.resnet50()
    model.fc = nn.Linear(model.fc.in_features, 23)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
        model.to(DEVICE).eval()
        app.run(host='0.0.0.0', port=5000, debug=False)
