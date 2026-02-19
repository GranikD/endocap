import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Укажите путь к распакованному датасету HyperKvasir
DATA_ROOT = './hyperkvasir-dataset'
CSV_FILE = os.path.join(DATA_ROOT, 'kvasir_v2.csv')  # Или название вашего csv файла с метками
IMAGE_FOLDER = os.path.join(DATA_ROOT, 'labeled-images')

BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Используется устройство: {DEVICE}")


# 1. ПОДГОТОВКА ДАТАСЕТА

class KvasirDataset(Dataset):
    def init(self, dataframe, root_dir, transform=None):
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform

        # Создаем словарь классов (строки -> числа)
        self.classes = self.dataframe['label'].unique()  # Убедитесь, что колонка называется 'label' или 'class'
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

    def len(self):
        return len(self.dataframe)

    def getitem(self, idx):
        # Получаем имя файла (в HyperKvasir это часто просто ID или путь)
        img_name = str(self.dataframe.iloc[idx]['image_id'])
        # Добавляем расширение, если его нет в csv (проверьте ваш csv)
        if not img_name.endswith('.jpg'):
            img_name += '.jpg'

        img_path = os.path.join(self.root_dir, self.dataframe.iloc[idx]['folder_name'],
                                img_name)  # folder_name часто есть в csv

        # Загрузка изображения
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            # Заглушка на случай битого пути (для безопасности)
            image = Image.new('RGB', (224, 224))

        label_name = self.dataframe.iloc[idx]['label']
        label = self.class_to_idx[label_name]

        if self.transform:
            image = self.transform(image)

        return image, label


# 2. ЗАГРУЗКА И ОБРАБОТКА ДАННЫХ

# Читаем CSV
# Примечание: Вам нужно отфильтровать CSV, если вы ищете ТОЛЬКО опухоли.
# Например: df = df[df['label'].isin(['polyps', 'normal-z-line'])]
try:
    df = pd.read_csv(CSV_FILE)  # Убедитесь, что разделитель верный (запятая или точка с запятой)
except FileNotFoundError:
    print("CSV файл не найден. Создаем фиктивный DataFrame для примера кода.")
    data = {'image_id': ['test1', 'test2'], 'label': ['polyp', 'normal'], 'folder_name': ['polyps', 'normal']}
    df = pd.DataFrame(data)

# Разделение на train/val
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

# Аугментация изображений (Важно для медицинских данных!)
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = KvasirDataset(train_df, IMAGE_FOLDER, transform=train_transforms)
val_dataset = KvasirDataset(val_df, IMAGE_FOLDER, transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 3. СОЗДАНИЕ МОДЕЛИ
def get_model(num_classes):
    # Загружаем предобученную модель
    model = models.resnet18(pretrained=True)



    # Заменяем последний слой под количество наших классов
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model


num_classes = len(train_dataset.classes)
model = get_model(num_classes).to(DEVICE)

# 4. ОБУЧЕНИЕ

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


def train_one_epoch(epoch_index):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch_index + 1}/{NUM_EPOCHS}")

    for images, labels in loop:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        # Zero gradients
        optimizer.zero_grad()

        # Forward
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward
        loss.backward()
        optimizer.step()

        # Metrics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        loop.set_postfix(loss=loss.item())

    epoch_acc = 100 * correct / total
    print(f"Epoch {epoch_index + 1} Result: Loss: {running_loss / len(train_loader):.4f}, Acc: {epoch_acc:.2f}%")


def validate():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Validation Accuracy: {100 * correct / total:.2f}%")
    print("-" * 20)


# Запуск цикла обучения

print(f"Начинаем обучение на {num_classes} классов: {train_dataset.classes}")
for epoch in range(NUM_EPOCHS):
    train_one_epoch(epoch)
    validate()

    # Сохранение модели
    torch.save(model.state_dict(), 'gi_tumor_detector.pth')
    print("Модель сохранена как gi_tumor_detector.pth")
