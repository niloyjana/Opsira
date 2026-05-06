#  Opsira — Ocular Intelligence Dashboard

Opsira is a high-performance, hybrid AI diagnostic platform for rapid ocular screening. By combining classical Computer Vision (OpenCV) with Deep Learning (TensorFlow), Opsira provides a multi-layered analysis of eye health, including physical metrics, disease classification, and explainable AI heatmaps. 

---
 
## 🚀 Key Features

- **Hybrid Analysis Engine**: Combines physical metric extraction (OpenCV) with neural network classification (EfficientNetB0).
- **Physical Biomarkers (Problem 1)**: Real-time calculation of ocular metrics:
  - 🔴 **Redness**: Detects inflammation and vascular congestion.
  - 🟡 **Yellowness**: Identifies early indicators of Jaundice (Icterus).
  - 🌫️ **Cloudiness**: Detects lens opacity related to Cataracts. 
- **Explainable AI (Grad-CAM)**: Generates visual heatmaps to show exactly which areas of the eye influenced the AI's decision.
- **Feature-Model Boosting (Problem 2)**: A unique logic layer that adjusts neural confidence scores based on extracted physical biomarkers for higher diagnostic accuracy.
- **Clinical Protocols**: Automatically suggests next steps and severity levels for detected conditions.

---

## 📂 Project Structure

```text
Opsira/
├── frontend/             # Modern Dashboard UI
│   ├── index.html
│   ├── style.css
│   └── script.js
├── backend/              # AI Logic & API
│   ├── predict.py        # Main Backend & Hybrid Pipeline (Working)
│   ├── app.py            # Legacy entry point
│   └── train_model.py    # Training scripts
├── model/                # Pre-trained Weights (H5/Keras)
└── requirements.txt      # System Dependencies
```

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/niloyjana/Opsira.git
cd Opsira
```

### 2. Environment Setup
We recommend using a virtual environment (Python 3.10+):
```bash
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the Platform
Start the integrated backend server (it also serves the frontend):
```bash
python backend/predict.py
```
Visit **[http://localhost:5000](http://localhost:5000)** in your browser.

---

## 🧠 Diagnostic Pipeline

1. **Preprocessing**: Image is normalized and resized for the EfficientNet backbone.
2. **Metric Extraction**: OpenCV filters calculate redness, jaundice, and cloudiness indices.
3. **Neural Inference**: The model predicts probabilities for Healthy, Cataract, Jaundice, and Red Eye.
4. **Boosting Layer**: If physical metrics (e.g., high cloudiness) contradict or strongly support the model, the probabilities are dynamically adjusted.
5. **Visualization**: A Grad-CAM heatmap is overlaid on the original image to provide clinical transparency.

---

## 🧪 Custom Training

You can retrain Opsira on your own clinical datasets:
1. Place your images in `dataset/<category_name>/`.
2. Run:
   ```bash
   python backend/train_model.py --epochs 30
   ```

---

Developed with ❤️ for Ocular Health Research.
