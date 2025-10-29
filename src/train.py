import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
import yaml, os, json
from datetime import datetime

params = yaml.safe_load(open("params.yaml"))
train_p = params["train"]
data_p = params["data"]

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_data = datasets.ImageFolder(data_p["train_path"], transform=transform)
train_loader = DataLoader(train_data, batch_size=train_p["batch_size"], shuffle=True)

# ======= ВЫБОР МОДЕЛИ =======
def build_model(name, num_classes):
    if name == "smallcnn":
        return nn.Sequential(
            nn.Conv2d(3, 32, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 54 * 54, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    elif name == "resnet18":
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif name == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    else:
        raise ValueError("Unknown model")

model = build_model(train_p["model_name"], len(train_data.classes)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=train_p["lr"])

# ======= Обучение =======
for epoch in range(train_p["epochs"]):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{train_p['epochs']}, Loss: {total_loss/len(train_loader):.4f}")

os.makedirs("models", exist_ok=True)
model_path = f"models/{train_p['model_name']}.pth"
torch.save(model.state_dict(), model_path)
print(f"✅ Model saved to {model_path}")

# Запишем метаинфо для DVC metrics
info = {"model": train_p["model_name"], "final_loss": total_loss / len(train_loader)}
os.makedirs("metrics", exist_ok=True)
with open("metrics/train_metrics.json", "w") as f:
    json.dump(info, f)
