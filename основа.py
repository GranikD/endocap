import cv2
import numpy as np
import torch
from albumentations import Compose, Resize, Normalize, HorizontalFlip
from monai.networks.nets import UNet

# Предобработка изображений
transform = Compose([
    Resize(256, 256),
    HorizontalFlip(p=0.5),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Модель U-Net для сегментации опухолей (2 класса: фон/опухоль)
model = UNet(
    spatial_dims=2,
    in_channels=3,
    out_channels=2,
    channels=(16, 32, 64, 128),
    strides=(2, 2, 2),
    num_res_units=2
).to(device)

# Загрузка предобученной модели (замените на вашу .pth)
model.load_state_dict(torch.load('unet_capsule.pth', map_location=device))
model.eval()


# Функция инференса на кадре
def detect_tumor(frame):
    input_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    augmented = transform(image=input_img)['image']
    input_tensor = torch.from_numpy(augmented).permute(2, 0, 1).unsqueeze(0).float().to(device)

    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()

    # Постобработка: маска опухоли
    mask = (pred == 1).astype(np.uint8) * 255
    return mask


# Обработка видео с капсулы (реал-тайм)
cap = cv2.VideoCapture('capsule_video.mp4')  # или 0 для камеры

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    mask = detect_tumor(frame)

    # Наложение маски на кадр
    frame_overlay = frame.copy()
    frame_overlay[mask > 0] = [0, 0, 255]  # Красный для опухоли

    cv2.imshow('Endoscopy Tumor Detection', frame_overlay)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
