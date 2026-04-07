"""
Unified Crop Disease Detection - Flask Web Application
Supports: Sugarcane (YOLOv8), Rice Severity (EfficientNet-B0), Wheat Disease (ResNet-18)
"""

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from pathlib import Path
from PIL import Image
from io import BytesIO
import base64
import os
import torch
import torch.nn as nn

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tiff"}
MAX_FILE_SIZE = 16 * 1024 * 1024

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Preprocessing (shared by rice & wheat) ────────────────────────────────────
from torchvision import transforms

clf_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ── Sugarcane YOLO models ──────────────────────────────────────────────────────
sugarcane_detection = None
sugarcane_segmentation = None
sugarcane_modes = []

try:
    from ultralytics import YOLO
    _det_path = Path("sugarcane/models/yolov8.pt")
    if _det_path.exists():
        sugarcane_detection = YOLO(str(_det_path))
        sugarcane_modes.append("detection")
        print("✓ Sugarcane detection model loaded")
    else:
        print(f"✗ Sugarcane detection model not found at {_det_path}")
except Exception as e:
    print(f"✗ Sugarcane detection load error: {e}")

try:
    _seg_path = Path("sugarcane/models/yolov8_seg.pt")
    if _seg_path.exists():
        sugarcane_segmentation = YOLO(str(_seg_path))
        sugarcane_modes.append("segmentation")
        print("✓ Sugarcane segmentation model loaded")
    else:
        print(f"✗ Sugarcane segmentation model not found at {_seg_path}")
except Exception as e:
    print(f"✗ Sugarcane segmentation load error: {e}")

# ── Rice severity model (EfficientNet-B0) ─────────────────────────────────────
rice_model = None
rice_classes = None

# Preferred class labels for the Rice model (ordered by class index).
# If you have an exact class order from training, put it in `rice_classes.json`
# as a JSON array and it will be loaded automatically.
RICE_CLASS_NAME_PRESETS = {
    9: [
        "Bacterial Blight Healthy",
        "Bacterial Blight Mild",
        "Bacterial Blight Severe",
        "Blast Healthy",
        "Blast Mild",
        "Blast Severe",
        "Brown Spot Healthy",
        "Brown Spot Mild",
        "Brown Spot Severe",
    ],
    12: [
        "Bacterial Blight Healthy",
        "Bacterial Blight Mild",
        "Bacterial Blight Severe",
        "Blast Healthy",
        "Blast Mild",
        "Blast Severe",
        "Brown Spot Healthy",
        "Brown Spot Mild",
        "Brown Spot Severe",
        "Tungro Healthy",
        "Tungro Mild",
        "Tungro Severe",
    ],
}

RICE_SEVERITY = {
    "healthy": 0,
    "mild": 25,
    "severe": 75,
}

try:
    import timm
    import json
    _rice_path = Path("rice_severity_model.pth")
    if _rice_path.exists():
        state = torch.load(_rice_path, map_location=device)
        _n = state["classifier.weight"].shape[0]
        rice_model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=_n)
        rice_model.load_state_dict(state)
        rice_model = rice_model.to(device)
        rice_model.eval()

        # Try loading exact class names from a local metadata file first.
        # File format: ["Class A", "Class B", ...]
        meta_path = Path("rice_classes.json")
        if meta_path.exists():
            try:
                loaded = json.loads(meta_path.read_text())
                if isinstance(loaded, list) and len(loaded) == _n:
                    rice_classes = [str(x) for x in loaded]
            except Exception:
                rice_classes = None

        # Fall back to known presets when metadata is not available.
        if rice_classes is None:
            rice_classes = RICE_CLASS_NAME_PRESETS.get(_n)

        print(f"✓ Rice model loaded ({_n} classes)")
        if rice_classes:
            print("✓ Rice class labels loaded")
        else:
            print("⚠ Rice class labels not found; using generic names")
    else:
        print(f"✗ Rice model not found at {_rice_path}")
except Exception as e:
    print(f"✗ Rice model load error: {e}")

# ── Wheat disease model (ResNet-18) ───────────────────────────────────────────
wheat_model = None

WHEAT_CLASSES = [
    "Brown Rust",
    "Healthy",
    "Loose Smut",
    "Mildew",
    "Yellow Rust",
]

WHEAT_INFO = {
    "Brown Rust":  {"severity": "high",   "color": "#b45309", "recommendation": "Apply fungicide (triazole/strobilurin). Remove infected leaves."},
    "Healthy":     {"severity": "none",   "color": "#10b981", "recommendation": "No action needed. Continue regular monitoring."},
    "Loose Smut":  {"severity": "high",   "color": "#7c3aed", "recommendation": "Use certified smut-free seed next season. Destroy infected heads."},
    "Mildew":      {"severity": "medium", "color": "#6366f1", "recommendation": "Improve air circulation. Apply fungicide if widespread."},
    "Yellow Rust": {"severity": "high",   "color": "#d97706", "recommendation": "Apply systemic fungicide immediately. Monitor spread closely."},
}

# ── Pesticide / chemical-composition database ─────────────────────────────────
PESTICIDE_DB = {
    # Sugarcane – generic classes produced by YOLOv8
    "disease": [
        {
            "product": "Propiconazole 25% EC",
            "active_ingredient": "Propiconazole",
            "chemical_group": "Triazole",
            "dosage": "1 ml / L water",
            "application": "Foliar spray; 2–3 rounds at 14-day intervals",
            "mode_of_action": "Demethylation inhibitor – blocks C14-ergosterol biosynthesis in fungal cell membranes",
        },
        {
            "product": "Carbendazim 50% WP",
            "active_ingredient": "Carbendazim",
            "chemical_group": "Benzimidazole",
            "dosage": "1 g / L water",
            "application": "Foliar spray at 10-day intervals",
            "mode_of_action": "Inhibits β-tubulin polymerisation; disrupts fungal cell division",
        },
        {
            "product": "Copper Oxychloride 50% WP",
            "active_ingredient": "Copper oxychloride (Cu₂(OH)₂Cl₂)",
            "chemical_group": "Inorganic copper compound",
            "dosage": "3 g / L water",
            "application": "Foliar spray; avoid high-heat conditions",
            "mode_of_action": "Cu²⁺ ions denature fungal enzymes and disrupt cell membranes",
        },
    ],
    "insect": [
        {
            "product": "Imidacloprid 17.8% SL",
            "active_ingredient": "Imidacloprid",
            "chemical_group": "Neonicotinoid",
            "dosage": "0.5 ml / L water",
            "application": "Foliar spray or soil drench",
            "mode_of_action": "Agonist of nicotinic acetylcholine receptors; causes irreversible nerve stimulation in insects",
        },
        {
            "product": "Chlorpyrifos 20% EC",
            "active_ingredient": "Chlorpyrifos",
            "chemical_group": "Organophosphate",
            "dosage": "2 ml / L water",
            "application": "Soil drench or foliar spray",
            "mode_of_action": "Inhibits acetylcholinesterase causing nervous system failure in insects",
        },
        {
            "product": "Fipronil 5% SC",
            "active_ingredient": "Fipronil",
            "chemical_group": "Phenylpyrazole",
            "dosage": "2 ml / L water",
            "application": "Soil drench at planting or stem-base spray",
            "mode_of_action": "Blocks GABA-gated chloride channels causing excessive CNS stimulation in insects",
        },
    ],
    # Rice
    "Bacterial Blight": [
        {
            "product": "Streptomycin 70% + Tetracycline 30% SP",
            "active_ingredient": "Streptomycin sulfate + Tetracycline HCl",
            "chemical_group": "Aminoglycoside + Tetracycline antibiotic",
            "dosage": "0.5 g / L water",
            "application": "Foliar spray; 2 rounds 7 days apart at disease onset",
            "mode_of_action": "Inhibits bacterial protein synthesis at the 30S ribosomal subunit",
        },
        {
            "product": "Kasugamycin 3% SL",
            "active_ingredient": "Kasugamycin",
            "chemical_group": "Aminoglycoside antibiotic",
            "dosage": "2 ml / L water",
            "application": "Foliar spray from tillering stage onwards",
            "mode_of_action": "Inhibits aminoacyl-tRNA binding; disrupts Xanthomonas oryzae protein synthesis",
        },
        {
            "product": "Copper Hydroxide 77% WP",
            "active_ingredient": "Copper hydroxide [Cu(OH)₂]",
            "chemical_group": "Inorganic copper compound",
            "dosage": "2 g / L water",
            "application": "Foliar spray; avoid application before rain",
            "mode_of_action": "Cu²⁺ ions disrupt bacterial membrane integrity and deactivate enzymes",
        },
    ],
    "Blast": [
        {
            "product": "Tricyclazole 75% WP",
            "active_ingredient": "Tricyclazole",
            "chemical_group": "Triazolo-pyrimidine",
            "dosage": "0.6 g / L water",
            "application": "Foliar spray at booting and heading stages",
            "mode_of_action": "Inhibits DHN-melanin biosynthesis; prevents appressorium penetration by Magnaporthe oryzae",
        },
        {
            "product": "Isoprothiolane 40% EC",
            "active_ingredient": "Isoprothiolane",
            "chemical_group": "Dithiolane",
            "dosage": "1.5 ml / L water",
            "application": "Foliar spray or root-zone treatment",
            "mode_of_action": "Inhibits fatty acid biosynthesis and melanin formation in Magnaporthe oryzae",
        },
        {
            "product": "Azoxystrobin 23% SC",
            "active_ingredient": "Azoxystrobin",
            "chemical_group": "Strobilurin (QoI fungicide)",
            "dosage": "1 ml / L water",
            "application": "Foliar spray at 10–14 day intervals",
            "mode_of_action": "Inhibits mitochondrial respiration at the Q₀ site of cytochrome bc₁ complex",
        },
    ],
    "Brown Spot": [
        {
            "product": "Mancozeb 75% WP",
            "active_ingredient": "Mancozeb (Mn/Zn ethylene bis-dithiocarbamate)",
            "chemical_group": "Dithiocarbamate",
            "dosage": "2.5 g / L water",
            "application": "Foliar spray at 7–10 day intervals; up to 3 applications",
            "mode_of_action": "Multi-site inhibitor; disrupts SH-containing enzymes in Helminthosporium oryzae",
        },
        {
            "product": "Iprodione 50% WP",
            "active_ingredient": "Iprodione",
            "chemical_group": "Dicarboximide",
            "dosage": "2 g / L water",
            "application": "Foliar spray at first symptom appearance",
            "mode_of_action": "Inhibits spore germination and hyphal growth via osmotic signal transduction disruption",
        },
        {
            "product": "Propiconazole 25% EC",
            "active_ingredient": "Propiconazole",
            "chemical_group": "Triazole",
            "dosage": "1 ml / L water",
            "application": "Foliar spray; 2 applications at 14-day intervals",
            "mode_of_action": "Demethylation inhibitor (DMI); blocks ergosterol biosynthesis",
        },
    ],
    "Tungro": [
        {
            "product": "Imidacloprid 17.8% SL",
            "active_ingredient": "Imidacloprid",
            "chemical_group": "Neonicotinoid (leafhopper vector control)",
            "dosage": "0.5 ml / L foliar | 1 ml / L seedling root dip for 12 h",
            "application": "Controls rice green leafhopper (Nephotettix virescens) – the Tungro virus vector",
            "mode_of_action": "Blocks nicotinic acetylcholine receptors; prevents virus spread by eliminating vector insect",
        },
        {
            "product": "Carbofuran 3% G",
            "active_ingredient": "Carbofuran",
            "chemical_group": "Carbamate",
            "dosage": "25 kg / ha soil incorporation",
            "application": "Soil incorporation before transplanting",
            "mode_of_action": "Systemic ACh-esterase inhibitor; kills leafhopper vectors in soil and water",
        },
    ],
    # Wheat
    "Brown Rust": [
        {
            "product": "Propiconazole 25% EC",
            "active_ingredient": "Propiconazole",
            "chemical_group": "Triazole",
            "dosage": "1 ml / L water",
            "application": "Foliar spray at disease onset; repeat after 14 days",
            "mode_of_action": "Inhibits ergosterol biosynthesis (C14-demethylation) in Puccinia triticina",
        },
        {
            "product": "Tebuconazole 25.9% EC",
            "active_ingredient": "Tebuconazole",
            "chemical_group": "Triazole",
            "dosage": "1 ml / L water",
            "application": "Foliar spray at flag leaf to heading stage",
            "mode_of_action": "DMI fungicide – inhibits C14α-demethylation in sterol biosynthesis",
        },
        {
            "product": "Trifloxystrobin 25% + Tebuconazole 50% WG",
            "active_ingredient": "Trifloxystrobin + Tebuconazole",
            "chemical_group": "Strobilurin + Triazole combination",
            "dosage": "0.5 g / L water",
            "application": "Foliar spray; protective + curative dual action",
            "mode_of_action": "QoI mitochondrial respiration inhibition + DMI ergosterol biosynthesis inhibition",
        },
    ],
    "Yellow Rust": [
        {
            "product": "Propiconazole 25% EC",
            "active_ingredient": "Propiconazole",
            "chemical_group": "Triazole",
            "dosage": "1 ml / L water",
            "application": "Spray immediately at first stripe-pustule symptom",
            "mode_of_action": "Inhibits ergosterol biosynthesis in Puccinia striiformis f. sp. tritici",
        },
        {
            "product": "Azoxystrobin 23% SC",
            "active_ingredient": "Azoxystrobin",
            "chemical_group": "Strobilurin",
            "dosage": "1 ml / L water",
            "application": "Preventive foliar spray before disease pressure builds",
            "mode_of_action": "Inhibits mitochondrial respiration at cytochrome bc₁ complex",
        },
        {
            "product": "Mancozeb 75% WP",
            "active_ingredient": "Mancozeb",
            "chemical_group": "Dithiocarbamate",
            "dosage": "2.5 g / L water",
            "application": "Protective contact foliar spray",
            "mode_of_action": "Multi-site contact fungicide; inhibits enzyme activity during uredospore germination",
        },
    ],
    "Mildew": [
        {
            "product": "Sulphur 80% WP",
            "active_ingredient": "Elemental Sulphur (S₈)",
            "chemical_group": "Inorganic sulphur",
            "dosage": "3 g / L water",
            "application": "Foliar spray early morning or evening; avoid above 32 °C",
            "mode_of_action": "Interferes with electron transport and oxidative phosphorylation in Blumeria graminis",
        },
        {
            "product": "Triadimefon 25% WP",
            "active_ingredient": "Triadimefon",
            "chemical_group": "Triazole",
            "dosage": "0.5 g / L water",
            "application": "Foliar spray at powdery mildew onset",
            "mode_of_action": "Systemic DMI – curative action against Blumeria graminis f. sp. tritici",
        },
        {
            "product": "Azoxystrobin 23% SC",
            "active_ingredient": "Azoxystrobin",
            "chemical_group": "Strobilurin",
            "dosage": "1 ml / L water",
            "application": "Preventive + curative foliar spray",
            "mode_of_action": "Inhibits spore germination and mycelial growth via mitochondrial respiration disruption",
        },
    ],
    "Loose Smut": [
        {
            "product": "Carboxin 37.5% + Thiram 37.5% DS",
            "active_ingredient": "Carboxin + Thiram",
            "chemical_group": "Oxathiin fungicide + Dithiocarbamate (seed treatment)",
            "dosage": "3 g / kg seed",
            "application": "Seed treatment before sowing",
            "mode_of_action": "Carboxin inhibits succinate dehydrogenase in Ustilago tritici; Thiram is a multi-site contact protectant",
        },
        {
            "product": "Tebuconazole 2% DS",
            "active_ingredient": "Tebuconazole",
            "chemical_group": "Triazole (seed treatment)",
            "dosage": "1.5 g / kg seed",
            "application": "Dry seed treatment before sowing",
            "mode_of_action": "Systemic seed treatment; inhibits ergosterol biosynthesis in Ustilago tritici",
        },
        {
            "product": "Carbendazim 50% WP",
            "active_ingredient": "Carbendazim",
            "chemical_group": "Benzimidazole (seed treatment)",
            "dosage": "2.5 g / kg seed | soak seeds for 6 h",
            "application": "Seed treatment before sowing",
            "mode_of_action": "Inhibits β-tubulin polymerisation; systemic control of seed-borne smut",
        },
    ],
    "Healthy": [],
}


def get_pesticides(disease_key: str) -> list:
    """Return pesticide suggestions for a given disease or detection class."""
    if disease_key in PESTICIDE_DB:
        return PESTICIDE_DB[disease_key]
    key_lower = disease_key.lower()
    for k, v in PESTICIDE_DB.items():
        if k.lower() == key_lower:
            return v
    return []


try:
    from torchvision import models as tv_models
    _wheat_path = Path("wheat_disease_resnet18.pth")
    if _wheat_path.exists():
        state = torch.load(_wheat_path, map_location=device)
        _n = state["fc.weight"].shape[0]
        wheat_model = tv_models.resnet18(weights=None)
        wheat_model.fc = nn.Linear(wheat_model.fc.in_features, _n)
        wheat_model.load_state_dict(state)
        wheat_model = wheat_model.to(device)
        wheat_model.eval()
        print(f"✓ Wheat model loaded ({_n} classes)")
    else:
        print(f"✗ Wheat model not found at {_wheat_path}")
except Exception as e:
    print(f"✗ Wheat model load error: {e}")

# ── Helpers ───────────────────────────────────────────────────────────────────
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def pil_to_base64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _resolve_rice_class_name(class_name: str) -> int:
    name = class_name.lower()
    if "healthy" in name:
        return 0
    elif "mild" in name:
        return 25
    elif "severe" in name:
        return 75
    return 50


def _parse_rice_label(class_name: str):
    """Split a rice class label into disease + severity parts for display."""
    normalized = class_name.replace("_", " ").strip()
    lowered = normalized.lower()

    if "healthy" in lowered:
        severity_label = "Healthy"
    elif "mild" in lowered:
        severity_label = "Mild"
    elif "severe" in lowered:
        severity_label = "Severe"
    else:
        severity_label = "Unknown"

    if severity_label.lower() in lowered:
        disease = normalized.lower().replace(severity_label.lower(), "").strip().title()
    else:
        disease = normalized.title()

    if not disease:
        disease = "Unknown"

    return disease, severity_label


# ── Sugarcane CLASS_INFO ──────────────────────────────────────────────────────
SUGARCANE_CLASS_INFO = {
    "healthy": {
        "color": "#10b981", "icon": "✓", "description": "Healthy sugarcane tissue",
        "severity": "low",
        "recommendation": "No action needed. Continue regular monitoring and maintenance.",
    },
    "disease": {
        "color": "#ef4444", "icon": "⚠", "description": "Disease detected",
        "severity": "high",
        "recommendation": "Apply appropriate fungicide immediately. Isolate affected plants if severe.",
    },
    "insect": {
        "color": "#f59e0b", "icon": "⚡", "description": "Insect pest detected",
        "severity": "medium",
        "recommendation": "Apply targeted insecticide. Consider biological pest control methods.",
    },
}

# ── Sugarcane processing ──────────────────────────────────────────────────────
def process_sugarcane(filepath, mode="segmentation", conf=0.25):
    import cv2
    import numpy as np

    if sugarcane_segmentation is None:
        return {"success": False, "error": "Sugarcane precise model is not available."}

    model = sugarcane_segmentation
    mode = "segmentation"

    results = model(filepath, conf=conf)
    ann = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)
    annotated_pil = Image.fromarray(ann)

    boxes = results[0].boxes
    names = results[0].names

    if len(boxes) == 0:
        analysis = {
            "status": "healthy",
            "message": "No issues detected. Your sugarcane appears healthy.",
            "total": 0,
            "detections": [],
            "recommendations": ["Continue regular monitoring", "Check again in 1–2 weeks"],
            "model_type": mode,
        }
    else:
        counts, confs = {}, {}
        for box in boxes:
            cls_name = names[int(box.cls[0])]
            c = float(box.conf[0])
            counts[cls_name] = counts.get(cls_name, 0) + 1
            confs.setdefault(cls_name, []).append(c)

        details = []
        sev = "low"
        for cls_name, cnt in counts.items():
            info = SUGARCANE_CLASS_INFO.get(cls_name, {})
            avg_c = sum(confs[cls_name]) / cnt
            details.append({
                "class": cls_name, "count": cnt,
                "confidence": round(avg_c * 100, 1),
                "color": info.get("color", "#6b7280"),
                "icon": info.get("icon", "•"),
                "description": info.get("description", ""),
                "severity": info.get("severity", "low"),
                "recommendation": info.get("recommendation", "Monitor closely"),
                "pesticides": get_pesticides(cls_name),
            })
            if info.get("severity") == "high":
                sev = "high"
            elif info.get("severity") == "medium" and sev != "high":
                sev = "medium"

        recs = []
        if "disease" in counts:
            recs += ["Apply appropriate fungicide or bactericide", "Isolate severely affected plants"]
        if "insect" in counts:
            recs += ["Apply targeted pest control", "Consider integrated pest management (IPM)"]
        if not recs:
            recs = ["Continue regular monitoring"]

        status = "critical" if sev == "high" else ("warning" if sev == "medium" else "healthy")
        analysis = {
            "status": status,
            "message": f"Detected {len(boxes)} issue(s) in the image.",
            "total": len(boxes),
            "detections": details,
            "recommendations": recs,
            "model_type": mode,
        }

    return {"success": True, "image": pil_to_base64(annotated_pil), "analysis": analysis}


# ── Rice processing ───────────────────────────────────────────────────────────
def process_rice(filepath):
    if rice_model is None:
        return {"success": False, "error": "Rice model not available."}

    img = Image.open(filepath).convert("RGB")
    original_b64 = pil_to_base64(img)
    tensor = clf_transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = rice_model(tensor)
        probs = torch.softmax(output, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
        n = probs.shape[0]

    # Build dynamic class names
    def _name(i):
        if rice_classes and i < len(rice_classes):
            return rice_classes[i]
        return f"Class_{i}"

    pred_name = _name(pred_idx)
    confidence = round(probs[pred_idx].item() * 100, 1)
    severity = _resolve_rice_class_name(pred_name)
    disease_name, severity_label = _parse_rice_label(pred_name)

    top3_vals, top3_idx = torch.topk(probs, k=min(3, n))
    top3 = [{"name": _name(int(i)), "confidence": round(float(v) * 100, 1)}
            for v, i in zip(top3_vals, top3_idx)]

    name_lower = pred_name.lower()
    if "severe" in name_lower:
        status, rec = "critical", "Immediate treatment required. Apply fungicide and isolate affected area."
    elif "mild" in name_lower:
        status, rec = "warning", "Apply preventative fungicide. Increase monitoring frequency."
    else:
        status, rec = "healthy", "No action needed. Continue regular monitoring."

    analysis = {
        "status": status,
        "class": pred_name,
        "disease": disease_name,
        "severity_label": severity_label,
        "confidence": confidence,
        "severity_pct": severity,
        "recommendations": [rec],
        "top3": top3,
        "pesticides": get_pesticides(disease_name),
    }

    return {"success": True, "image": original_b64, "analysis": analysis}


# ── Wheat processing ──────────────────────────────────────────────────────────
def process_wheat(filepath):
    if wheat_model is None:
        return {"success": False, "error": "Wheat model not available."}

    img = Image.open(filepath).convert("RGB")
    original_b64 = pil_to_base64(img)
    tensor = clf_transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = wheat_model(tensor)
        probs = torch.softmax(output, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
        n = probs.shape[0]

    def _name(i):
        return WHEAT_CLASSES[i] if i < len(WHEAT_CLASSES) else f"Class_{i}"

    pred_name = _name(pred_idx)
    confidence = round(probs[pred_idx].item() * 100, 1)
    info = WHEAT_INFO.get(pred_name, {"severity": "unknown", "color": "#6b7280", "recommendation": "Consult an agronomist."})

    top3_vals, top3_idx = torch.topk(probs, k=min(3, n))
    top3 = [{"name": _name(int(i)), "confidence": round(float(v) * 100, 1)}
            for v, i in zip(top3_vals, top3_idx)]

    sev = info["severity"]
    status = "critical" if sev == "high" else ("warning" if sev == "medium" else "healthy")

    analysis = {
        "status": status,
        "class": pred_name,
        "confidence": confidence,
        "color": info["color"],
        "recommendations": [info["recommendation"]],
        "top3": top3,
        "pesticides": get_pesticides(pred_name),
    }

    return {"success": True, "image": original_b64, "analysis": analysis}


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/analyze/sugarcane", methods=["POST"])
def analyze_sugarcane():
    try:
        files = request.files.getlist("files")
        if not files or all(f.filename == "" for f in files):
            return jsonify({"success": False, "error": "No files uploaded"}), 400

        mode = "segmentation"
        conf = float(request.form.get("conf_threshold", 0.25))

        batch = []
        for file in files:
            if not file.filename or not allowed_file(file.filename):
                batch.append({"success": False, "error": f"Invalid file: {file.filename}", "filename": file.filename})
                continue
            fname = secure_filename(file.filename)
            fpath = os.path.join(app.config["UPLOAD_FOLDER"], fname)
            file.save(fpath)
            result = process_sugarcane(fpath, mode, conf)
            result["filename"] = file.filename
            os.remove(fpath)
            batch.append(result)

        return jsonify({"success": True, "results": batch})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/analyze/rice", methods=["POST"])
def analyze_rice():
    try:
        files = request.files.getlist("files")
        if not files or all(f.filename == "" for f in files):
            return jsonify({"success": False, "error": "No files uploaded"}), 400

        batch = []
        for file in files:
            if not file.filename or not allowed_file(file.filename):
                batch.append({"success": False, "error": f"Invalid file: {file.filename}", "filename": file.filename})
                continue
            fname = secure_filename(file.filename)
            fpath = os.path.join(app.config["UPLOAD_FOLDER"], fname)
            file.save(fpath)
            result = process_rice(fpath)
            result["filename"] = file.filename
            os.remove(fpath)
            batch.append(result)

        return jsonify({"success": True, "results": batch})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/analyze/wheat", methods=["POST"])
def analyze_wheat():
    try:
        files = request.files.getlist("files")
        if not files or all(f.filename == "" for f in files):
            return jsonify({"success": False, "error": "No files uploaded"}), 400

        batch = []
        for file in files:
            if not file.filename or not allowed_file(file.filename):
                batch.append({"success": False, "error": f"Invalid file: {file.filename}", "filename": file.filename})
                continue
            fname = secure_filename(file.filename)
            fpath = os.path.join(app.config["UPLOAD_FOLDER"], fname)
            file.save(fpath)
            result = process_wheat(fpath)
            result["filename"] = file.filename
            os.remove(fpath)
            batch.append(result)

        return jsonify({"success": True, "results": batch})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "sugarcane_modes": sugarcane_modes,
        "rice_model": rice_model is not None,
        "wheat_model": wheat_model is not None,
        "device": str(device),
    })


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🌾 Unified Crop Disease Detection System")
    print("=" * 60)
    print(f"  Sugarcane : {', '.join(sugarcane_modes) or 'unavailable'}")
    print(f"  Rice      : {'loaded' if rice_model else 'unavailable'}")
    print(f"  Wheat     : {'loaded' if wheat_model else 'unavailable'}")
    print(f"  Device    : {device}")
    print("\n📍 Open your browser to: http://localhost:6005")
    print("=" * 60 + "\n")
    app.run(debug=False, host="0.0.0.0", port=6005, use_reloader=False)
