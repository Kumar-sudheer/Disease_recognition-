"""
Rice Leaf Disease Severity Prediction
Architecture: EfficientNet-B0 (timm)
Usage: python riceSeverityPredict.py <image_path>
"""

import sys
import torch
import torch.nn as nn
import timm
from torchvision import transforms
from PIL import Image

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_PATH = "rice_severity_model.pth"

# Class names must be in the same alphabetical order as the training dataset folders.
# Run the training script once and check the printed "Classes:" line to verify.
# Example order from the Kaggle "severity-based-rice-leaf-diseases-dataset":
CLASS_NAMES = [
    "Bacterialblight_Healthy",
    "Bacterialblight_Mild",
    "Bacterialblight_Severe",
    "Blast_Healthy",
    "Blast_Mild",
    "Blast_Severe",
    "Brownspot_Healthy",
    "Brownspot_Mild",
    "Brownspot_Severe",
    "Tungro_Healthy",
    "Tungro_Mild",
    "Tungro_Severe",
]

# Severity score per class name keyword (matches training idx_to_severity logic)
def get_severity(class_name: str) -> int:
    name = class_name.lower()
    if "healthy" in name:
        return 0
    elif "mild" in name:
        return 25
    elif "severe" in name:
        return 75
    return 50

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
num_classes = state_dict["classifier.weight"].shape[0]
print(f"Detected {num_classes} classes in saved model.")

if len(CLASS_NAMES) != num_classes:
    print(
        f"WARNING: CLASS_NAMES has {len(CLASS_NAMES)} entries but model has "
        f"{num_classes} outputs. Update CLASS_NAMES to match training folders."
    )

model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=num_classes)
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
    severity = get_severity(class_name)

    print(f"\nImage       : {image_path}")
    print(f"Prediction  : {class_name}")
    print(f"Confidence  : {confidence:.2f}%")
    print(f"Severity    : {severity}%")

    # Top-3 predictions
    top3 = torch.topk(probabilities, k=min(3, num_classes))
    print("\nTop-3 predictions:")
    for score, idx in zip(top3.values, top3.indices):
        name = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f"Class_{idx}"
        print(f"  {name:<35} {score.item()*100:.2f}%")

    return class_name, confidence, severity

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python riceSeverityPredict.py <image_path>")
        sys.exit(1)

    predict(sys.argv[1])
