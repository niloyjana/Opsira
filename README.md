# 👁️ VisionGuard AI

## Project Structure
```
visionguard/
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
├── backend/
│   ├── app.py
│   ├── predict.py
│   └── train_model.py
├── dataset/
│   ├── healthy/
│   ├── cataract/
│   ├── red_eye/
│   └── jaundice/
└── requirements.txt
```

## Run (2 steps)

### Step 1 — Start the backend
```
cd visionguard
pip install -r requirements.txt
python backend/app.py
```

### Step 2 — Open the frontend
Just open `frontend/index.html` in your browser. That's it!

## Train the model (optional)
```
python backend/train_model.py --data_dir ./dataset --epochs 30
```
