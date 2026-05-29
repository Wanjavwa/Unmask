# Unmask — Dependency / Version Report

A snapshot was saved to `docs/pip-freeze-backend.txt` (torch, fastapi, opencv, etc.).

Regenerate anytime:

```powershell
cd unmask-backend
.\venv\Scripts\pip.exe freeze > ..\docs\pip-freeze-backend.txt
cd ..\fairness_model
pip freeze > pip-freeze-fairness.txt
```

## Backend (`unmask-backend/requirements.txt`)

| Package | Role |
|---------|------|
| fastapi | HTTP API |
| uvicorn | ASGI server |
| torch | Inference |
| torchvision | ResNet18 fairness head |
| pillow | Image I/O |
| python-multipart | File uploads |
| opencv-python-headless | Face detection |
| efficientnet-pytorch | EffB4 backbone |

**Risk:** Unpinned versions can change softmax/numerics slightly across torch releases. Pin for production.

## Fairness training (`fairness_model/requirements.txt`)

| Package | Role |
|---------|------|
| torch, torchvision | Training / ResNet18 |
| scikit-learn | Metrics |
| facenet-pytorch | Optional MTCNN (training) |
| opencv-python-headless | Utils |

## Mobile (`mobile/package.json`)

| Package | Version (approx.) |
|---------|-------------------|
| expo | ~54.0.33 |
| react | 19.1.0 |
| react-native | 0.81.5 |

## Environment consistency

- Use **same venv** for backend: `unmask-backend\venv`
- Run backend with **venv Python**: `.\venv\Scripts\python.exe -m uvicorn ...`
- GPU vs CPU: scores can differ slightly; set `UNMASK_DETERMINISTIC=1` for repeatable CPU tests
