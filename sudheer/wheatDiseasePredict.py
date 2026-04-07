"""
Wheat Leaf Disease Prediction
Architecture: ResNet-18 (torchvision)
Usage: python wheatDiseasePredict.py <image_path>
"""

import sys
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_PATH = "wheat_disease_resnet18.pth"

# Class names must be in the same alphabetical order as the training dataset folders.
# Run the training script once and check the printed "Classes:" line to verify.
# The Kaggle "wheat-leaf-disease-dataset" by khanaamer has these 5 classes:
CLASS_NAMES = [
    "Brown Rust",
    "Healthy",
    "Loose Smut",
    "Mildew",
    "Yellow Rust",
]

# ── Preprocessing (same as training, without augmentation) ────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ── Load model ────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

state_dict = torch.load(MODEL_PATH, map_location=device)

# Auto-detect number of classes from saved weights
num_classes = state_dict["fc.weight"].shape[0]
print(f"Detected {num_classes} classes in saved model.")

if len(CLASS_NAMES) != num_classes:
    print(
        f"WARNING: CLASS_NAMES has {len(CLASS_NAMES)} entries but model has "
        f"{num_classes} outputs. Update CLASS_NAMES to match training folders."
    )

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

print("Model loaded successfully.")

# ── Prediction function ───────────────────────────────────────────────────────
def predict(image_path: str):
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        probabilities = torch.softmax(output, dim=1)[0]
        pred_idx = torch.argmax(probabilities).item()

    if pred_idx < len(CLASS_NAMES):
        class_name = CLASS_NAMES[pred_idx]
    else:
        class_name = f"Class_{pred_idx}"

    confidence = probabilities[pred_idx].item() * 100

    print(f"\nImage       : {image_path}")
    print(f"Prediction  : {class_name}")
    print(f"Confidence  : {confidence:.2f}%")

    # Top-3 predictions
    top3 = torch.topk(probabilities, k=min(3, num_classes))
    print("\nTop-3 predictions:")
    for score, idx in zip(top3.values, top3.indices):
        name = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f"Class_{idx}"
        print(f"  {name:<20} {score.item()*100:.2f}%")

    return class_name, confidence

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python wheatDiseasePredict.py <image_path>")
        sys.exit(1)

    predict(sys.argv[1])
