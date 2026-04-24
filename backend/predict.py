"""
Opsira AI — Backend Prediction Pipeline
Fixed for Problem 1 & 2 + Static File Serving
"""

import os
import io
import json
import base64
import datetime
import numpy as np
import cv2
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import tensorflow as tf

app = Flask(__name__)
CORS(app)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "opsira_model.h5")
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")

# Global Baselines for Normalization
HEALTHY_BASELINES = {
    "redness":    {"mean": 0.18, "std": 0.06},
    "yellowness": {"mean": 0.10, "std": 0.05},
    "cloudiness": {"mean": 0.12, "std": 0.05},
    "brightness": {"mean": 0.50, "std": 0.10},
    "contrast":   {"mean": 0.55, "std": 0.10},
}

DISEASE_METADATA = {
    "cataract": {
        "label": "Possible Cataract",
        "advice": "Cataracts cause clouding of the lens. Early surgical treatment is highly effective.",
        "protocol": "Please consult an ophthalmologist promptly.",
        "severity": "HIGH"
    },
    "healthy": {
        "label": "Healthy Eye",
        "advice": "Your eye appears healthy. Maintain regular check-ups.",
        "protocol": "Maintain current ocular hygiene.",
        "severity": "LOW"
    },
    "jaundice": {
        "label": "Jaundice (Icterus)",
        "advice": "Yellowing of the sclera indicates potential liver or bile duct issues.",
        "protocol": "🚨 Immediate medical evaluation of liver function required.",
        "severity": "HIGH"
    },
    "red_eye": {
        "label": "Conjunctivitis / Red Eye",
        "advice": "Red eye may indicate infection or inflammation.",
        "protocol": "🩺 Seek medical advice within 24-48 hours.",
        "severity": "MODERATE"
    }
}

# Global Model Variable
_model = None
_classes = ["cataract", "healthy", "jaundice", "red_eye"]

def load_opsira_model():
    global _model
    if _model is None:
        if os.path.exists(MODEL_PATH):
            _model = tf.keras.models.load_model(MODEL_PATH)
            print(f">>> Opsira Model loaded from {MODEL_PATH}")
        else:
            print(f"!!! Warning: Model not found at {MODEL_PATH}. Running in simulation mode.")
    return _model

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def extract_metrics(img_bgr):
    """Problem 1: OpenCV Metric Extraction"""
    # Normalize to 0-1 for calculation
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    r, g, b = img_rgb[:,:,0], img_rgb[:,:,1], img_rgb[:,:,2]
    
    # Raw Metrics
    redness_raw    = float(np.mean(np.clip(r - (g + b) / 2, 0, 1)))
    yellowness_raw = float(np.mean(np.clip((r + g) / 2 - b, 0, 1)))
    
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    laplacian_variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    cloudiness_raw = float(1 / (1 + laplacian_variance / 100))
    
    brightness_raw = float(np.mean(img_rgb))
    contrast_raw   = float(np.std(img_rgb))
    
    raw = {
        "redness": redness_raw,
        "yellowness": yellowness_raw,
        "cloudiness": cloudiness_raw,
        "brightness": brightness_raw,
        "contrast": contrast_raw
    }
    
    # Normalized Metrics (Z-score -> Sigmoid)
    norm = {}
    for key, val in raw.items():
        base = HEALTHY_BASELINES[key]
        z_score = (val - base["mean"]) / (base["std"] + 1e-6)
        norm[key] = round(float(sigmoid(z_score) * 100), 1)
        
    return raw, norm

def generate_simulated_heatmap(img_bgr, raw_metrics):
    """Generates a content-aware synthetic heatmap for Simulation Mode"""
    h, w = img_bgr.shape[:2]
    heatmap = np.zeros((h, w), dtype=np.float32)
    
    # 1. Cloudiness (Cataract) -> Center Focus (Pupil/Lens)
    if raw_metrics["cloudiness"] > 0.25:
        cv2.circle(heatmap, (w//2, h//2), min(w, h)//4, 1.0, -1)
        
    # 2. Yellowness (Jaundice) -> Sclera Focus (Sides)
    if raw_metrics["yellowness"] > 0.15:
        # Create masks for the left and right sides of the eye
        cv2.rectangle(heatmap, (0, 0), (int(w*0.3), h), 0.8, -1)
        cv2.rectangle(heatmap, (int(w*0.7), 0), (w, h), 0.8, -1)

    # 3. Redness (Inflammation) -> Pixel-level highlight
    if raw_metrics["redness"] > 0.15:
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, np.array([0, 40, 40]), np.array([10, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([170, 40, 40]), np.array([180, 255, 255]))
        red_mask = cv2.add(mask1, mask2).astype(np.float32) / 255.0
        heatmap = cv2.addWeighted(heatmap, 0.5, red_mask, 0.5, 0)

    # Final Polish: Blur and Normalize
    heatmap = cv2.GaussianBlur(heatmap, (max(w,h)//10|1, max(w,h)//10|1), 0)
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    
    return cv2.resize(heatmap, (224, 224))

def apply_hybrid_boosting(probs, raw_metrics):
    """Problem 2: Feature-Model Boosting Logic (Refined)"""
    boosted = probs.copy()
    
    c = raw_metrics["cloudiness"]
    y = raw_metrics["yellowness"]
    r = raw_metrics["redness"]
    
    # 1. Jaundice (High Precision Color Feature)
    if y > 0.18:
        y_boost = 0.20 * (y / 0.25)
        boosted["jaundice"] += min(y_boost, 0.35)
        boosted["cataract"] -= 0.10
        boosted["healthy"]  -= 0.15
        
    # 2. Red Eye (High Precision Color Feature)
    if r > 0.22:
        r_boost = 0.15 * (r / 0.30)
        boosted["red_eye"] += min(r_boost, 0.30)
        boosted["healthy"] -= 0.10
        
    # 3. Cataract (Low Precision - Generic Blur)
    # Only boost if it's NOT looking like Jaundice or Red Eye
    if c > 0.45:
        if y < 0.22 and r < 0.25:
            boosted["cataract"] += 0.15
        else:
            # If color features are present, cloudiness is likely just image blur
            boosted["cataract"] += 0.02 
            
    # Clamp and Re-normalize
    boosted = {k: max(0.0001, v) for k, v in boosted.items()}
    total = sum(boosted.values())
    return {k: v / total for k, v in boosted.items()}

def get_gradcam(img_array, model):
    """Generates Grad-CAM heatmap base64 string"""
    try:
        # Find last conv layer
        last_conv_layer_name = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D) or "conv" in layer.name.lower():
                last_conv_layer_name = layer.name
                break
        
        if not last_conv_layer_name: return None
        
        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
        )
        
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, np.argmax(predictions[0])]
            
        output = conv_outputs[0]
        grads = tape.gradient(loss, conv_outputs)[0]
        weights = tf.reduce_mean(grads, axis=(0, 1))
        cam = np.dot(output, weights)
        
        cam = cv2.resize(cam, (224, 224))
        cam = np.maximum(cam, 0)
        heatmap = (cam - cam.min()) / (cam.max() - cam.min() + 1e-10)
        return heatmap
    except:
        return None

def apply_heatmap(img_bgr, heatmap):
    if heatmap is None: return img_bgr
    heatmap = np.uint8(255 * heatmap)
    colormap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    result = cv2.addWeighted(img_bgr, 0.6, colormap, 0.4, 0)
    return result

# --- STATIC ROUTES ---
@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, 'index.html')

@app.route("/<path:path>")
def serve_static(path):
    return send_from_directory(FRONTEND_DIR, path)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

# --- API ROUTES ---
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400
        
        file = request.files["image"]
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return jsonify({"error": "Invalid image format"}), 400
        
        # 1. Extract Metrics (Problem 1)
        raw_metrics, norm_metrics = extract_metrics(img_bgr)
        
        # 2. Neural Prediction
        model = load_opsira_model()
        img_input = cv2.resize(img_bgr, (224, 224))
        img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
        img_tensor = np.expand_dims(img_input.astype(np.float32) / 255.0, 0)
        
        if model:
            preds = model.predict(img_tensor, verbose=0)[0]
            probs = {_classes[i]: float(preds[i]) for i in range(4)}
            heatmap = get_gradcam(img_tensor, model)
        else:
            probs = {"cataract": 0.1, "healthy": 0.7, "jaundice": 0.1, "red_eye": 0.1}
            heatmap = generate_simulated_heatmap(img_bgr, raw_metrics)
            
        # 3. Hybrid Boosting (Problem 2)
        boosted_probs = apply_hybrid_boosting(probs, raw_metrics)
        top_class = max(boosted_probs, key=boosted_probs.get)
        meta = DISEASE_METADATA[top_class]
        
        # 4. Prepare Images (Original & Heatmap)
        h, w = img_bgr.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h)) if heatmap is not None else None
        
        _, buffer_orig = cv2.imencode(".jpg", img_bgr)
        orig_b64 = base64.b64encode(buffer_orig).decode("utf-8")
        
        heatmap_img = apply_heatmap(img_bgr, heatmap_resized)
        _, buffer_heat = cv2.imencode(".jpg", heatmap_img)
        heat_b64 = base64.b64encode(buffer_heat).decode("utf-8")
        
        # 5. Final Response
        response = {
            "prediction":      top_class,
            "confidence":      round(float(boosted_probs[top_class] * 100), 1),
            "severity":        meta["severity"],
            "label":           meta["label"],
            "clinical_advice": meta["advice"],
            "protocol":        meta["protocol"],
            "probabilities": {k: round(v * 100, 1) for k, v in boosted_probs.items()},
            "metrics":         norm_metrics,
            "radar": {
                "current":  [norm_metrics["redness"], norm_metrics["yellowness"], norm_metrics["cloudiness"], norm_metrics["contrast"], norm_metrics["brightness"]],
                "baseline": [20, 20, 20, 20, 20],
                "labels":   ["Redness", "Yellow", "Cloud", "Contrast", "Bright"]
            },
            "original_image": orig_b64,
            "heatmap_image":  heat_b64
        }
        
        return jsonify(response)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    load_opsira_model()
    app.run(debug=True, port=5000)
