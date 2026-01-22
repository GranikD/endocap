import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

dataset = datasets.ImageFolder(
    root="HyperKvasir/labeled-images",
    transform=transform
)

train_loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4
)

num_classes = len(dataset.classes)
print(dataset.classes)
