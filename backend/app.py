"""
Opsira — Flask Backend
Run: python app.py
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import base64
import io
from PIL import Image
from predict import predict_image, get_deviation_data, generate_text_report

app = Flask(__name__)
CORS(app)

# Serve Frontend
@app.route("/")
def index():
    return send_from_directory('../frontend', 'index.html')

@app.route("/<path:path>")
def serve_static(path):
    return send_from_directory('../frontend', path)

@app.route("/api/status")
def status():
    return jsonify({"status": "Opsira AI backend running"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Check if file part exists
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        if file:
            image = Image.open(file.stream).convert("RGB")

            # Run prediction
            result = predict_image(image)
            deviations = get_deviation_data(result["features"])
            report = generate_text_report(result)

            return jsonify({
                "predicted_class": result["predicted_class"],
                "confidence":       round(result["confidence"] * 100, 1),
                "probabilities":    {k: round(v * 100, 1) for k, v in result["probabilities"].items()},
                "risk_level":       result["risk_level"],
                "disease_info":     result["disease_info"],
                "features":         result["features"],
                "deviations":       deviations,
                "demo_mode":        result.get("demo_mode", False),
                "report":           report,
                "heatmap":          result["heatmap_image"],
            })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":

    print("\n>>> Opsira backend started!")
    print("    Open frontend/index.html in your browser\n")
    app.run(debug=True, port=5000)
