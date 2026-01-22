import torchvision.models as models
import torch.nn as nn

model = models.efficientnet_b0(pretrained=True)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    num_classes
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
