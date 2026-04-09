
import os
import time
import threading
import torch
import torch.nn as nn
import numpy as np
import cv2
from datetime import datetime
from torchvision import models, transforms
from PIL import Image
from flask import Flask, request, render_template_string, send_from_directory, session, redirect, url_for, jsonify
from flask_sqlalchemy import SQLAlchemy
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# ==========================================
# 1. ГЛОБАЛЬНАЯ КОНФИГУРАЦИЯ
# ==========================================
MODEL_PATH = r'kvasir_resnet50_26classes.pth'
BLUETOOTH_FOLDER = r'C:\Users\Unicum_Student\Desktop\Bluetooth_Photos'
STATIC_FOLDER = 'static'

# Создание структуры папок
for path in [STATIC_FOLDER, BLUETOOTH_FOLDER]:
    if not os.path.exists(path):
        os.makedirs(path)

app = Flask(__name__)
app.secret_key = 'caduceus_ultra_full_v9'

# ==========================================
# 2. БАЗА ДАННЫХ (ЧАТ)
# ==========================================
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///caduceus_chat.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    role = db.Column(db.String(20), nullable=False)  # 'doctor' или 'patient'
    text = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)


with app.app_context():
    db.create_all()

# Хранилище для последнего результата (синхронизация Bluetooth -> Веб)
latest_scan_data = {
    "diag": "Ожидание данных...",
    "conf": 0,
    "img": None,
    "timestamp": 0
}

# ==========================================
# 3. МОДУЛЬ НЕЙРОСЕТИ (AI)
# ==========================================
class_names = [
    'Accessory tools', 'Angiectasia', 'Barrett\'s esophagus', 'Blood in lumen',
    'Cecum', 'Colon diverticula', 'Colorectal cancer', 'Duodenal bulb',
    'Dyed-lifted-polyps', 'Dyed-resection-margins', 'Erythema', 'Esophageal varices',
    'Esophagitis', 'Gastroesophageal_junction_normal z-line', 'Ileocecal valve',
    'Mucosal inflammation large bowel', 'Normal esophagus',
    'Normal mucosa and vascular pattern in the colon', 'Normal stomach',
    'Polyps', 'Pylorus', 'Resected polyps', 'Resection margins',
    'Retroflex rectum', 'Small bowel_terminal ileum', 'Ulcer'
]

NORMAL_CLASSES = [
    'Cecum', 'Duodenal bulb', 'Gastroesophageal_junction_normal z-line',
    'Ileocecal valve', 'Normal esophagus', 'Normal stomach', 'Pylorus',
    'Retroflex rectum', 'Small bowel_terminal ileum',
    'Normal mucosa and vascular pattern in the colon'
]

translate_dict = {
    'Accessory tools': 'Медицинские инструменты',
    'Angiectasia': 'Ангиэктазия (сосуды)',
    'Barrett\'s esophagus': 'Пищевод Барретта',
    'Blood in lumen': 'Кровь в просвете',
    'Cecum': 'Слепая кишка (Норма)',
    'Colon diverticula': 'Дивертикулы кишечника',
    'Colorectal cancer': 'Колоректальный рак',
    'Duodenal bulb': 'Луковица 12-перстной кишки (Норма)',
    'Dyed-lifted-polyps': 'Полип (окрашен)',
    'Dyed-resection-margins': 'Края резекции (окрашены)',
    'Erythema': 'Эритема (покраснение)',
    'Esophageal varices': 'Варикоз пищевода',
    'Esophagitis': 'Эзофагит',
    'Gastroesophageal_junction_normal z-line': 'Z-линия (Норма)',
    'Ileocecal valve': 'Илеоцекальный клапан (Норма)',
    'Mucosal inflammation large bowel': 'Колит (воспаление)',
    'Normal esophagus': 'Пищевод (Норма)',
    'Normal mucosa and vascular pattern in the colon': 'Слизистая кишечника (Норма)',
    'Normal stomach': 'Желудок (Норма)',
    'Polyps': 'Полипы',
    'Pylorus': 'Привратник желудка (Норма)',
    'Resected polyps': 'Место удаления полипа',
    'Resection margins': 'Границы резекции',
    'Retroflex rectum': 'Прямая кишка (Ретрофлексия)',
    'Small bowel_terminal ileum': 'Терминальный отдел (Норма)',
    'Ulcer': 'Язва'
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None


def load_ai_model():
    global model
    model = models.resnet50()
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    if os.path.exists(MODEL_PATH):
        # weights_only=True для безопасности (убирает FutureWarning)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
        model.to(DEVICE).eval()
        print(">>> [AI] Модель загружена успешно.")


def perform_inference(image_path):
    global latest_scan_data
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    try:
        raw_img = Image.open(image_path).convert('RGB')
        orig_w, orig_h = raw_img.size
        input_tensor = preprocess(raw_img).unsqueeze(0).to(DEVICE)

        # Подготовка Grad-CAM (поиск зоны патологии)
        activation_maps = []

        def hook(m, i, o):
            activation_maps.append(o)

        handle = model.layer4.register_forward_hook(hook)

        with torch.no_grad():
            output = model(input_tensor)
        handle.remove()

        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        conf, index = torch.max(probabilities, 0)
        cls_idx = index.item()
        found_class = class_names[cls_idx]

        # Визуализация (OpenCV)
        output_cv = cv2.cvtColor(np.array(raw_img), cv2.COLOR_RGB2BGR)

        if found_class not in NORMAL_CLASSES:
            weights = list(model.fc.parameters())[0].data
            f_map = activation_maps[0].squeeze(0)
            cam = torch.matmul(weights[cls_idx], f_map.reshape(2048, -1)).reshape(7, 7).cpu().numpy()
            cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)
            heatmap = cv2.resize(np.uint8(255 * cam), (orig_w, orig_h))
            _, thresh = cv2.threshold(heatmap, 155, 255, cv2.THRESH_BINARY)
            cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                best_cnt = max(cnts, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(best_cnt)
                cv2.rectangle(output_cv, (x, y), (x + w, y + h), (0, 255, 0), 4)
                cv2.putText(output_cv, "PATHOLOGY DETECTED", (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        file_name = f"result_{int(time.time())}.jpg"
        save_path = os.path.join(STATIC_FOLDER, file_name)
        cv2.imwrite(save_path, output_cv)

        # Обновляем состояние для фронтенда
        latest_scan_data = {
            "diag": translate_dict.get(found_class, found_class),
            "conf": round(conf.item() * 100, 2),
            "img": file_name,
            "timestamp": time.time()
        }
        return latest_scan_data
    except Exception as e:
        print(f">>> [AI Error]: {e}")
        return None


# ==========================================
# 4. BLUETOOTH МОНИТОРИНГ (WATCHDOG)
# ==========================================
class BluetoothFolderHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith(('.jpg', '.png', '.jpeg')):
            print(f">>> [Bluetooth] Новый файл обнаружен: {event.src_path}")
            time.sleep(1.5)  # Даем время системе завершить запись файла
            perform_inference(event.src_path)


def start_observer():
    handler = BluetoothFolderHandler()
    observer = Observer()
    observer.schedule(handler, BLUETOOTH_FOLDER, recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except:
        observer.stop()
    observer.join()


# ==========================================
# 5. ВЕБ-ИНТЕРФЕЙС (FLASK + HTML)
# ==========================================
HTML_PAGE = """
<!DOCTYPE html>
<html lang="ru" data-theme="light">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>КАДУЦЕЙ AI | Эндоскопия</title>
    <style>
        :root { --main: #2ecc71; --main-dark: #27ae60; }
        [data-theme="light"] { --bg: #f5f7f6; --card: #ffffff; --txt: #2c3e50; --brd: #ecf0f1; --m-me: #2ecc71; --m-ot: #f1f2f6; --t-me: #fff; }
        [data-theme="dark"] { --bg: #1a1a1a; --card: #2d2d2d; --txt: #ecf0f1; --brd: #404040; --m-me: #27ae60; --m-ot: #3d3d3d; --t-me: #fff; }

        body { font-family: 'Segoe UI', Tahoma, sans-serif; background: var(--bg); color: var(--txt); margin: 0; transition: 0.3s; }
        .wrapper { display: grid; grid-template-columns: 1fr 400px; gap: 20px; max-width: 1500px; margin: 20px auto; padding: 0 20px; }

        .header { text-align: center; padding: 20px; background: var(--card); border-bottom: 2px solid var(--main); margin-bottom: 20px; }
        .card { background: var(--card); border-radius: 15px; padding: 25px; border: 1px solid var(--brd); box-shadow: 0 4px 15px rgba(0,0,0,0.1); }

        /* ЧАТ */
        .chat-wrap { display: flex; flex-direction: column; height: 85vh; background: var(--card); border-radius: 15px; border: 1px solid var(--brd); overflow: hidden; }
        .chat-head { background: #2c3e50; color: #fff; padding: 15px; text-align: center; font-weight: bold; }
        .chat-msgs { flex: 1; padding: 20px; overflow-y: auto; display: flex; flex-direction: column; gap: 15px; }

        .message { max-width: 85%; padding: 12px 18px; border-radius: 20px; font-size: 0.95rem; position: relative; }
        .message.me { align-self: flex-end; background: var(--m-me); color: var(--t-me); border-bottom-right-radius: 4px; }
        .message.other { align-self: flex-start; background: var(--m-ot); color: var(--txt); border-bottom-left-radius: 4px; }
        .message small { display: block; font-size: 0.7rem; margin-bottom: 4px; opacity: 0.8; font-weight: bold; }

        .chat-input { padding: 20px; border-top: 1px solid var(--brd); display: flex; gap: 10px; }
        .chat-input input { flex: 1; padding: 12px; border-radius: 10px; border: 1px solid var(--brd); background: var(--bg); color: var(--txt); outline: none; }
        .chat-input button { background: var(--main); color: white; border: none; padding: 0 20px; border-radius: 10px; cursor: pointer; font-weight: bold; }

        /* ИНТЕРФЕЙС ИИ */
        .diag-badge { display: inline-block; padding: 12px 25px; background: rgba(46, 204, 113, 0.1); color: var(--main); border-radius: 50px; font-weight: bold; border: 2px solid var(--main); margin-bottom: 20px; font-size: 1.1rem; }
        .res-img { width: 100%; border-radius: 12px; border: 1px solid var(--brd); box-shadow: 0 5px 15px rgba(0,0,0,0.2); }

        .theme-switch { position: fixed; top: 20px; right: 20px; background: var(--card); border: 1px solid var(--brd); padding: 10px; border-radius: 50%; cursor: pointer; font-size: 1.5rem; z-index: 1000; }
        .role-selector { display: flex; justify-content: center; gap: 20px; margin-top: 100px; }
        .btn-role { padding: 30px 50px; border-radius: 20px; text-decoration: none; font-weight: bold; font-size: 1.2rem; color: #fff; transition: 0.3s; }
    </style>
</head>
<body>
    <button class="theme-switch" onclick="toggleTheme()" id="thIcon">🌙</button>

    {% if not session.get('role') %}
    <div class="header"><h1>КАДУЦЕЙ AI</h1><p>Интеллектуальный помощник врача-эндоскописта</p></div>
    <div class="role-selector">
        <a href="/set_role/doctor" class="btn-role" style="background: #3498db;">👨‍⚕️ Я ВРАЧ</a>
        <a href="/set_role/patient" class="btn-role" style="background: #e67e22;">👤 Я ПАЦИЕНТ</a>
    </div>
    {% else %}
    <div class="header">
        <h1>КАДУЦЕЙ AI</h1>
        <p>Вы вошли как: <b>{{ 'ВРАЧ' if session['role'] == 'doctor' else 'ПАЦИЕНТ' }}</b> | <a href="/logout" style="color: red;">Выйти</a></p>
    </div>

    <div class="wrapper">
        <div class="card">
            <div id="diagArea">
                <div class="diag-badge" id="diagLabel">Ожидание первого снимка...</div>
            </div>

            <div id="imgContainer">
                <img id="mainImage" class="res-img" style="display: none;">
                <p id="placeholderText">Снимки от пациента или по Bluetooth появятся здесь автоматически.</p>
            </div>

            {% if session['role'] == 'patient' %}
            <div style="margin-top: 30px; padding: 20px; border: 2px dashed var(--main); border-radius: 15px; background: rgba(46, 204, 113, 0.05);">
                <h3>Загрузить снимок кишечника</h3>
                <form method="POST" enctype="multipart/form-data">
                    <input type="file" name="file" id="fileInp" style="display: none;" onchange="this.form.submit()">
                    <button type="button" onclick="document.getElementById('fileInp').click()" style="background: var(--main); color: white; border: none; padding: 15px 30px; border-radius: 10px; font-weight: bold; cursor: pointer; width: 100%;">ВЫБРАТЬ И ОТПРАВИТЬ ФАЙЛ</button>
                </form>
            </div>
            {% else %}
            <div style="margin-top: 30px; padding: 20px; background: rgba(0,0,0,0.05); border-radius: 10px;">
                <p>💡 <b>Совет для врача:</b> Снимки поступают в реальном времени. Если вы используете Bluetooth-эндоскоп, просто отправьте фото в папку <code>C:\Bluetooth_Photos</code>.</p>
            </div>
            {% endif %}
        </div>

        <div class="chat-wrap">
            <div class="chat-head">ЧАТ: {{ 'ПАЦИЕНТ' if session['role'] == 'doctor' else 'ЛЕЧАЩИЙ ВРАЧ' }}</div>
            <div class="chat-msgs" id="chatWin"></div>
            <form id="chatForm" class="chat-input">
                <input type="text" id="mText" placeholder="Введите сообщение..." required autocomplete="off">
                <button type="submit">ОТПРАВИТЬ</button>
            </form>
        </div>
    </div>
    {% endif %}

    <script>
        const myRole = "{{ session.get('role', '') }}";
        let lastSyncTime = 0;

        // 1. АВТОМАТИЧЕСКОЕ ПОЛУЧЕНИЕ СНИМКОВ (AJAX)
        function pollScanner() {
            fetch('/api/latest_scan')
                .then(r => r.json())
                .then(data => {
                    if (data.timestamp > lastSyncTime) {
                        lastSyncTime = data.timestamp;
                        document.getElementById('diagLabel').innerText = `ДИАГНОЗ: ${data.diag} (${data.conf}%)`;
                        const img = document.getElementById('mainImage');
                        img.src = "/static/" + data.img + "?t=" + new Date().getTime();
                        img.style.display = "block";
                        document.getElementById('placeholderText').style.display = "none";
                    }
                });
        }

        // 2. РАБОТА С ЧАТОМ
        let messageCount = 0;
        function updateChat() {
            fetch('/api/get_messages')
                .then(r => r.json())
                .then(data => {
                    if (data.length !== messageCount) {
                        const win = document.getElementById('chatWin');
                        win.innerHTML = data.map(m => `
                            <div class="message ${m.role === myRole ? 'me' : 'other'}">
                                <small>${m.role === 'doctor' ? '👨‍⚕️ ВРАЧ' : '👤 ПАЦИЕНТ'}</small>
                                ${m.text}
                            </div>
                        `).join('');
                        messageCount = data.length;
                        win.scrollTop = win.scrollHeight;
                    }
                });
        }

        if (myRole) {
            setInterval(pollScanner, 3000);
            setInterval(updateChat, 2000);

            document.getElementById('chatForm').onsubmit = (e) => {
                e.preventDefault();
                const inp = document.getElementById('mText');
                const fd = new FormData();
                fd.append('text', inp.value);
                fetch('/api/send_message', { method: 'POST', body: fd }).then(() => {
                    inp.value = '';
                    updateChat();
                });
            };
        }

        function toggleTheme() {
            const h = document.documentElement;
            const isDark = h.getAttribute('data-theme') === 'dark';
            const newT = isDark ? 'light' : 'dark';
            h.setAttribute('data-theme', newT);
            document.getElementById('thIcon').innerText = isDark ? '🌙' : '☀️';
            localStorage.setItem('cad_theme', newT);
        }
        document.documentElement.setAttribute('data-theme', localStorage.getItem('cad_theme') || 'light');
    </script>
</body>
</html>
"""


# ==========================================
# 6. API И МАРШРУТИЗАЦИЯ FLASK
# ==========================================
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST' and session.get('role') == 'patient':
        file = request.files.get('file')
        if file:
            temp_path = os.path.join(STATIC_FOLDER, "temp_upload.jpg")
            file.save(temp_path)
            perform_inference(temp_path)
    return render_template_string(HTML_PAGE)


@app.route('/api/latest_scan')
def get_latest_scan():
    return jsonify(latest_scan_result=latest_scan_data['diag'] if latest_scan_data['img'] else None, **latest_scan_data)


@app.route('/api/get_messages')
def get_messages():
    msgs = Message.query.order_by(Message.timestamp.asc()).all()
    return jsonify([{"role": m.role, "text": m.text} for m in msgs])


@app.route('/api/send_message', methods=['POST'])
def send_msg():
    role = session.get('role')
    text = request.form.get('text')
    if role and text:
        db.session.add(Message(role=role, text=text))
        db.session.commit()
    return '', 204


@app.route('/set_role/<role>')
def set_role(role):
    if role in ['doctor', 'patient']:
        session['role'] = role
    return redirect(url_for('index'))


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))


@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(STATIC_FOLDER, filename)


# ==========================================
# 7. ЗАПУСК ПРИЛОЖЕНИЯ
# ==========================================
if __name__ == "__main__":
    # Загрузка ИИ
    load_ai_model()

    # Запуск Bluetooth-мониторинга в отдельном потоке (daemon)
    threading.Thread(target=start_observer, daemon=True).start()

    # Запуск сервера
    app.run(host='0.0.0.0', port=1000, debug=False)

