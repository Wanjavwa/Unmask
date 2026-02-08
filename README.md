# Unmask: Equity-Centered Deepfake Detection (Mobile + FastAPI)

Unmask is a mobile deepfake verification tool designed to protect communities from synthetic media harm through an equity and racial justice lens.  
Unlike many detection systems that produce overconfident outputs, Unmask uses a **multi-model ensemble** and an **uncertainty layer** to reduce harmful misclassification—especially against people of color.

---

## ✨ Key Features

### Deepfake Detection (Backend)

- Upload an image and receive a deepfake prediction in real time
- Uses a multi-model ensemble for more stable results
- Produces structured detection outputs (label + confidence)

### Uncertainty Layer (Bias-Aware Output)

- Detects inconsistent model behavior and disproportionate readings
- Prevents overconfident "Real/Fake" results when predictions are unstable
- Returns an uncertainty margin instead of forcing absolute conclusions

### Transparent & Human-Readable Explanations

- Every prediction includes a plain-language explanation
- Confidence interpretation (High / Low)
- Built-in disclaimer communicating model limitations and bias risks

### Mobile App (React Native / Expo)

- Select an image from your phone gallery
- Share to verify from other apps
- Upload to backend automatically
- View result label, confidence score, explanation, and disclaimer in-app
- Elegant, accessible UI with real-time feedback

---

## 🧱 Project Structure

```
Unmask/
├── README.md
│
├── unmask-backend/              # FastAPI backend + deepfake inference
│   ├── app.py                   # FastAPI application & API endpoints
│   ├── model.py                 # Deepfake detection model logic
│   ├── requirements.txt          # Python dependencies
│   ├── run.bat                  # Windows batch runner
│   ├── DeepfakeBench/           # Benchmark & reference models
│   └── [model weights]          # Pre-trained detection models
│
├── fairness_model/              # Fairness-aware ML pipeline
│   ├── train.py                 # Training script for fairness model
│   ├── infer.py                 # Inference utilities
│   ├── requirements.txt          # ML dependencies
│   ├── src/                     # Core ML modules
│   │   ├── config.py            # Configuration
│   │   ├── model_builder.py     # Model architecture
│   │   ├── dataset_loader.py    # Data loading utilities
│   │   ├── metrics.py           # Fairness metrics
│   │   └── utils.py             # Helper functions
│   ├── models/                  # Trained model checkpoints
│   │   ├── fairness_head_best.pt
│   │   ├── fairness_head.pt
│   │   └── training_log.txt
│   └── data/                    # Training datasets
│       ├── fake/
│       └── real/
│
└── mobile/                      # React Native Expo frontend
    ├── App.js                   # Main app entry point
    ├── app.json                 # Expo configuration
    ├── package.json             # JavaScript dependencies
    ├── services/                # API & backend integration
    │   └── api.js               # Backend communication
    ├── src/                     # App utilities
    │   └── utils/
    │       └── shareHandler.js  # Share intent handling
    ├── android/                 # Android build files
    │   ├── app/
    │   │   ├── src/
    │   │   │   ├── main/        # Main Android manifest
    │   │   │   ├── debug/
    │   │   │   └── debugOptimized/
    │   │   └── build.gradle
    │   ├── build.gradle
    │   └── [gradle config files]
    └── [expo config files]
```

---

## ✅ Prerequisites

- Python 3.8+
- Node.js 16+
- npm
- Expo CLI (recommended)

---

## ⚙️ Setup & Run

### 1. Backend Setup (FastAPI)

```bash
cd unmask-backend
pip install -r requirements.txt
python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Backend will run at:

- **API Base:** `http://localhost:8000`
- **Swagger API Testing UI:** `http://localhost:8000/docs`

---

### 2. Frontend Setup (Expo Mobile App)

```bash
cd mobile
npm install
npx expo start
```

Expo will open a local development server and generate a QR code.  
Scan the QR code using the **Expo Go** app (Android/iOS) to run Unmask on your phone.

---

## 📡 API Overview

### POST /detect-image

Uploads an image and returns a deepfake detection response.

**Example Response:**

```json
{
  "label": "Likely Real",
  "confidence": 0.72,
  "explanation": "The face region was detected and analyzed. Confidence is high based on consistent model agreement.",
  "disclaimer": "Unmask provides probabilistic analysis and may be affected by dataset bias, lighting, and image quality. It should be used as decision-support, not definitive proof."
}
```

---

## 🎬 Demo

[Insert GIF demo here - showing app flow: image selection → upload → result display → share]

---

## ⚖️ Why Unmask Exists

Most deepfake detection systems are trained on datasets that are estimated to be over 80% white, meaning they often generalize poorly to darker skin tones.

Studies and fairness audits have shown that facial AI systems can make incorrect judgments of Black individuals at significantly higher rates—some evaluations reporting misclassification rates as high as 60% depending on task and dataset.

Unmask was built to reduce these harms by:

- Introducing uncertainty outputs instead of absolute claims
- Providing transparency through explanations and disclaimers
- Treating deepfake verification as decision-support, not truth declaration
- Auditing model performance across demographic groups

---

## 🛠️ Troubleshooting

| Issue                                       | Fix                                                                                            |
| ------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| Backend not reachable from mobile           | Make sure backend is running with `--host 0.0.0.0` and your phone is on the same WiFi network. |
| CORS errors                                 | Confirm CORS is enabled in FastAPI (check `allow_origins` in `app.py`).                        |
| Model weights missing                       | Ensure all required model files are in `unmask-backend/`.                                      |
| Expo not connecting                         | Restart Expo and try `npx expo start --tunnel`.                                                |
| Port already in use                         | Change port: `--port 8001` or stop existing processes.                                         |
| Android physical device can't reach backend | Update `ANDROID_PHYSICAL_IP` in `mobile/services/api.js` with your computer's local IP.        |

---

## 🚀 Future Work

- Support for deepfake video and audio verification
- Increase dataset to include other POC groups
- Clouad-deployable backend
- Web browser + social media extension and partnerships

---

## 📜 License

This project is intended for educational and hackathon demonstration purposes.  
Pretrained model weights remain under their original license.
