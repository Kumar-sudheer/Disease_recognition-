print("connected")

# !pip install kagglehub timm seaborn scikit-learn -q

import kagglehub
import os

path = kagglehub.dataset_download("isaacritharson/severity-based-rice-leaf-diseases-dataset")

print("Dataset Path:", path)

os.listdir(path)

import torch
import torch.nn as nn
import torch.optim as optim
import timm
import numpy as np
import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

dataset_path = os.path.join(path, "Leaf Disease Dataset")

train_dir = os.path.join(dataset_path, "train")
val_dir = os.path.join(dataset_path, "validation")

train_dataset = datasets.ImageFolder(train_dir, transform=transform)
val_dataset = datasets.ImageFolder(val_dir, transform=transform)

print("Classes:", train_dataset.classes)

print("Train images:", len(train_dataset))
print("Validation images:", len(val_dataset))

train_loader = DataLoader(train_dataset,
                          batch_size=32,
                          shuffle=True)

val_loader = DataLoader(val_dataset,
                        batch_size=32,
                        shuffle=False)

print("Train batches:", len(train_loader))
print("Validation batches:", len(val_loader))

severity_map = {
    "healthy":0,
    "mild":25,
    "severe":75
}

idx_to_severity = {}

for cls, idx in train_dataset.class_to_idx.items():

    name = cls.lower()

    if "healthy" in name:
        idx_to_severity[idx] = 0
    elif "mild" in name:
        idx_to_severity[idx] = 25
    elif "severe" in name:
        idx_to_severity[idx] = 75
    else:
        idx_to_severity[idx] = 50

print(idx_to_severity)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = timm.create_model("efficientnet_b0", pretrained=True)

model.classifier = nn.Linear(
    model.classifier.in_features,
    len(train_dataset.classes)
)

model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.0001)

train_losses = []

epochs = 25

for epoch in range(epochs):

    model.train()

    running_loss = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)

    train_losses.append(epoch_loss)

    print(f"Epoch {epoch+1}/{epochs} Loss: {epoch_loss:.4f}")

# training loss

plt.plot(train_losses)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title("Training Loss Curve")

plt.show()

# validation accuracy

model.eval()

correct = 0
total = 0

all_preds = []
all_labels = []

with torch.no_grad():

    for images, labels in val_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs,1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = 100 * correct / total

print("Validation Accuracy:", accuracy)

# classification report

print(classification_report(
    all_labels,
    all_preds,
    target_names=train_dataset.classes
))

# confusion matrix

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(10,8))

sns.heatmap(cm,
            annot=True,
            fmt="d",
            xticklabels=train_dataset.classes,
            yticklabels=train_dataset.classes)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

# visual text (fixed img dislay)

import random

mean = torch.tensor([0.485,0.456,0.406])
std = torch.tensor([0.229,0.224,0.225])

model.eval()

for i in range(10):

    idx = random.randint(0,len(val_dataset)-1)

    image, label = val_dataset[idx]

    img = image.unsqueeze(0).to(device)

    with torch.no_grad():

        output = model(img)

        pred = torch.argmax(output,1).item()

    img_show = image * std[:,None,None] + mean[:,None,None]

    plt.imshow(img_show.permute(1,2,0).clamp(0,1))

    plt.title(f"Pred: {train_dataset.classes[pred]}\nActual: {train_dataset.classes[label]}")

    plt.axis("off")

    plt.show()

# predict severity %

from PIL import Image

def predict_severity(img_path):

    img = Image.open(img_path).convert("RGB")

    img = transform(img).unsqueeze(0).to(device)

    model.eval()

    with torch.no_grad():

        output = model(img)

        pred = torch.argmax(output,1).item()

    severity = idx_to_severity[pred]

    print("Class:", train_dataset.classes[pred])
    print("Severity:", severity,"%")

    return severity

predict_severity("/content/bact_lf_bli.jpeg")

# save model

torch.save(model.state_dict(),"rice_severity_model.pth")