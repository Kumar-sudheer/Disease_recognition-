# Unified Crop Disease Detection App

A single Flask web app for crop image analysis across three model pipelines:

- Sugarcane: YOLOv8 object detection and segmentation
- Rice: EfficientNet-B0 severity classification
- Wheat: ResNet-18 disease classification

The app is served from the project root and currently configured to run on `http://localhost:6005`.

## Features

- One UI with crop tabs: Sugarcane, Rice, Wheat
- Drag-and-drop image upload
- Real-time inference via Flask API
- Model-specific result panels:
  - Sugarcane: detections, confidence, recommendations, annotated image
  - Rice: disease + severity level + severity percent + top-3 predictions
  - Wheat: disease + confidence + top-3 predictions

## Project Layout

```text
sudheer/
├── app.py
├── requirements.txt
├── README.md
├── templates/
│   └── index.html
├── static/
│   ├── style.css
│   └── script.js
├── uploads/                         # temp upload directory (auto-created)
├── rice_severity_model.pth
├── wheat_disease_resnet18.pth
├── sugarcane/
│   └── models/
│       ├── yolov8.pt
│       └── yolov8_seg.pt
├── riceSeverityTraining.py
├── wheatSeverityTraining.py
├── riceSeverityPredict.py
└── wheatDiseasePredict.py
```

## Requirements

- macOS, Linux, or Windows
- Python 3.10+ recommended
- `pip` or `pip3`
- Optional GPU acceleration (CUDA) if available

## 1. Setup

From project root:

```bash
cd /Users/menace/Downloads/development/sudheer
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If you do not use a virtual environment, install with `pip3` directly:

```bash
pip3 install -r requirements.txt
```

## 2. Verify Model Files

Make sure these files exist before starting:

- `rice_severity_model.pth`
- `wheat_disease_resnet18.pth`
- `sugarcane/models/yolov8.pt`
- `sugarcane/models/yolov8_seg.pt`

If a model is missing, the corresponding crop tab may return a model-not-available error.

## 3. Run the App

```bash
cd /Users/menace/Downloads/development/sudheer
python3 app.py
```

Open:

- `http://localhost:6005`

Health check endpoint:

- `http://localhost:6005/api/health`

## 4. How to Use the UI

### Sugarcane tab

1. Upload image
2. Select model type:
   - Object Detection (fast)
   - Instance Segmentation (precise)
3. Adjust confidence threshold
4. Click Analyze

Output includes annotated image, class counts, confidence, and recommendations.

### Rice tab

1. Upload rice leaf image
2. Click Analyze

Output includes:

- Disease name
- Severity level (Healthy / Mild / Severe)
- Severity percent (0/25/75)
- Top-3 predictions

### Wheat tab

1. Upload wheat leaf image
2. Click Analyze

Output includes:

- Disease name
- Confidence
- Top-3 predictions
- Recommendation

## Rice Class Label Mapping

Rice outputs use class labels loaded in this order:

1. `rice_classes.json` (if present, exact class order)
2. Built-in preset for 9-class or 12-class models
3. Fallback to `Class_<index>` if no mapping is available

If your model uses a custom class order, create `rice_classes.json` in root:

```json
[
  "Bacterial Blight Healthy",
  "Bacterial Blight Mild",
  "Bacterial Blight Severe",
  "Blast Healthy",
  "Blast Mild",
  "Blast Severe",
  "Brown Spot Healthy",
  "Brown Spot Mild",
  "Brown Spot Severe"
]
```

Important: the array order must exactly match training `class_to_idx`.

## API Endpoints

- `GET /` - web UI
- `GET /api/health` - service/model status
- `POST /api/analyze/sugarcane` - sugarcane inference
  - form-data: `file`, `model_type`, `conf_threshold`
- `POST /api/analyze/rice` - rice inference
  - form-data: `file`
- `POST /api/analyze/wheat` - wheat inference
  - form-data: `file`

## Common Issues

### `python: command not found`

Use `python3` instead:

```bash
python3 app.py
```

### Port already in use

App is set to `6005` in `app.py`. If needed, change:

```python
app.run(debug=False, host="0.0.0.0", port=6005, use_reloader=False)
```

### Favicon 404 in logs

`GET /favicon.ico 404` is harmless and does not affect app function.

### Model load error

Check:

- model file path exists
- file is not corrupted
- dependency versions installed from `requirements.txt`

## Development Notes

- Main backend: `app.py`
- Main frontend template: `templates/index.html`
- UI logic: `static/script.js`
- Styling: `static/style.css`

To stop the app, press `Ctrl + C` in the terminal running Flask.
