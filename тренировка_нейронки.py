import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from monai.data import CacheDataset
from monai.transforms import LoadImaged, EnsureChannelFirstd, ScaleIntensityd


class CapsuleDataset(Dataset):
    def __init__(self, image_paths, mask_paths):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = Compose([...])  # Ваши трансформации

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        mask = cv2.imread(self.mask_paths[idx], 0)
        augmented = self.transform(image=image, mask=mask)
        return augmented['image'], augmented['mask']


# Пример обучения
dataset = CacheDataset(data=[{'image': 'img.nii', 'label': 'mask.nii'}], transform=transforms)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

for epoch in range(50):
    for images, masks in dataloader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch}, Loss: {loss.item()}')

torch.save(model.state_dict(), 'unet_capsule.pth')
