import torch
from torchvision import models, transforms, datasets
import torch.nn as nn
import yaml, json
from sklearn.metrics import accuracy_score

params = yaml.safe_load(open("params.yaml"))
train_p = params["train"]
data_p = params["data"]
device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

val_data = datasets.ImageFolder("data/val_split", transform=transform)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=32)

# === модель ===
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
        model = models.resnet18()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif name == "efficientnet_b0":
        model = models.efficientnet_b0()
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model

model = build_model(train_p["model_name"], len(val_data.classes))
model.load_state_dict(torch.load(f"models/{train_p['model_name']}.pth"))
model.to(device)
model.eval()

# === оценка ===
all_preds, all_labels = [], []

with torch.no_grad():
    for imgs, labels in val_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs).argmax(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

acc = accuracy_score(all_labels, all_preds)
print(f"✅ Accuracy: {acc:.4f}")

metrics = {"accuracy": acc, "model": train_p["model_name"]}
with open("metrics/val_metrics.json", "w") as f:
    json.dump(metrics, f)
