# VisionGuard AI — Custom Training Guide 🧪

You can train this AI model on your own eye datasets to identify any specific diseases or conditions. Follow these steps:

## 1. Prepare Your Dataset
Organize your images into the `dataset/` folder. Create a new folder for each category (disease) you want the AI to learn.

**Example Structure:**
```text
dataset/
├── healthy/            <-- Put healthy eye images here
├── cataract/           <-- Put cataract images here
├── my_custom_disease/  <-- Put your own dataset images here
└── ...
```

*   **Format:** JPG, PNG, or WebP.
*   **Quantity:** For best results, aim for at least **50–100 images per category**.

## 2. Start Training
Open your terminal/command prompt in the project root and run:

```bash
# Install dependencies first
pip install -r requirements.txt

# Run the training script
python backend/train_model.py --epochs 30
```

### What happens during training?
1.  **Class Detection:** The script automatically detects your folders (e.g., `healthy`, `my_custom_disease`).
2.  **Learning:** It uses an **EfficientNetB0** neural network to learn visual patterns from your images.
3.  **Optimization:** It saves the "knowledge" into `model/visionguard_model.keras`.

## 3. Verify & Identify
Once training is finished:
1.  Restart the backend: `python backend/app.py`
2.  Open the frontend and upload a new image.
3.  The system will now calculate the score (confidence) and identify the disease based on **your specific data**.

---
**💡 Pro Tip:** If you have many images, you can increase `--epochs` (e.g., `--epochs 50`) for a more accurate model, but it will take longer to train.
