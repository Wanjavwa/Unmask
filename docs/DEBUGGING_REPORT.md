# Unmask Detection Pipeline — Debugging Report

**Date:** 2026-04-30  
**Scope:** Full audit of image deepfake detection (no video pipeline in codebase)

---

## Executive summary

Scores became inconsistent because of **multiple intentional pipeline changes** stacked together, not a single broken weight file. The main regressions were:

1. **Ensemble composition changed** — from 2-model average (EffB4 + Xception) to 3-model weighted sum with fairness at only **10% weight**, so the fairness head could not correct biased main detectors on Black faces.
2. **Confidence semantics changed** — the API/UI used **P(fake)** as “Confidence %”, while earlier behavior used **decision strength** `|P(fake) − 0.5| × 2`. A 25% fake probability displayed as “25% confidence” looks like low confidence in the verdict, but it is actually low fake probability (more likely real).
3. **Label thresholds drifted** — spec used 0.65 / 0.35; code had 0.6 / 0.4 at one point.
4. **Calibration** — `_calibrate()` maps `(p − 0.05) / 0.9` into [0.01, 0.99], shifting raw ensemble values.

**Measured failure (before fix):** Known real UTKFace Black image → `Likely deepfake`, prob_fake ≈ 0.65 (effb4=0.86, xception=0.55, fairness=0.02).

---

## Pipeline trace (actual implementation)

| Step | Implementation | Notes |
|------|----------------|-------|
| Input | Single image upload (`POST /detect-image`) | **No video / frame extraction** in repo |
| Face detect | OpenCV Haar cascade, largest face | Same crop for all models |
| Fallback | Center square crop if no face | |
| EffB4 / Xception preprocess | 256×256, mean/std = 0.5 | DeepfakeBench style |
| Fairness preprocess | 224×224, ImageNet mean/std | ResNet18 |
| Inference | Softmax class index **1 = fake** | Consistent across models |
| Ensemble | Weighted sum + fairness fusion | See fixes below |
| Calibration | `_calibrate(raw_prob)` | |
| Labels | Thresholds on calibrated prob_fake | |
| API | `label`, `confidence`, `prob_fake`, `explanation` | |
| UI | Bar = prob_fake; text shows both metrics | |

---

## Root causes (detailed)

### 1. Fairness under-weighted (critical)

Fairness ResNet18 was trained on Black real/fake data and correctly scored a real sample at **~2% fake**, but EffB4/Xception scored **~86% / ~55% fake**. With weights 0.45 / 0.45 / 0.10, the ensemble stayed **~65% fake** → wrong label.

### 2. Confidence display mismatch

| Metric | Meaning | Old UI (approx.) | Broken UI |
|--------|---------|------------------|-----------|
| `prob_fake` | P(AI-generated) | N/A or separate | Shown as “Confidence” |
| `decision_confidence` | How sure the verdict is | `abs(p−0.5)×2` | Not shown |

### 3. No video pipeline

README mentions future video support; **only static images** are implemented. Do not debug “frame sampling” — it does not exist.

### 4. Model weights

- EffB4 / Xception: `unmask-backend/DeepfakeBench/weights/*.pth` — load with `strict=False` (many keys); OK if ≤20 missing.
- Fairness: `fairness_model/models/fairness_head_best.pt` — remapped to `backbone.*` keys; loads strict.

**Retraining:** Fairness model retrained on 43 fake + 239 real images; not required for this fix, but more fake diversity would help fairness head.

---

## Fixes applied

1. **Fairness-aware fusion** — When main detectors strongly disagree with fairness (e.g. fairness &lt; 0.25 and main avg &gt; 0.55), blend 50/50 toward fairness.
2. **Weights** — Default 0.40 / 0.40 / 0.20 (EffB4 / Xception / fairness).
3. **API** — Returns `prob_fake` and `confidence` (decision strength) separately.
4. **UI** — Shows “Fake probability” and “Decision confidence” separately; bar uses prob_fake.
5. **Thresholds** — `THRESHOLD_FAKE = 0.65`, `THRESHOLD_REAL = 0.35`.
6. **Diagnostics** — `UNMASK_DEBUG=1` enables structured pipeline logs (`diagnostics.py`).
7. **Determinism** — `UNMASK_DETERMINISTIC=1` sets seeds and cudnn deterministic mode.
8. **Tests** — `unmask-backend/tests/test_detection_pipeline.py` regression tests.

---

## Before vs after (verified on disk samples)

**Real:** `fairness_model/data/real/1_0_1_20161219160115237.jpg`

| | Before fix | After fix |
|---|------------|-----------|
| effb4 | ~0.86 | 0.865 |
| xception | ~0.55 | 0.549 |
| fairness | ~0.02 | 0.024 |
| prob_fake | ~0.65 | **0.351** |
| label | Likely deepfake | **Uncertain** |
| decision_confidence | (shown as 65% fake) | **0.30** |

**Fake:** `fairness_model/data/fake/openart-image__Vcu-xcx_1770431564188_raw.jpg`

| | After fix |
|---|-----------|
| prob_fake | **0.841** |
| label | Likely deepfake |
| effb4 / xception / fairness | 0.92 / 0.95 / 0.28 |

---

## Model evaluation (fairness training log)

From last `fairness_model/models/training_log.txt` (43 fake, 239 real):

- Val accuracy: 98.25%
- Val F1 (fake): 0.89
- Confusion matrix (val): [[52, 0], [1, 4]] — 1 fake missed, 0 false positives on real in val

Fairness head alone is strong on its val set; ensemble must let it influence final scores.

---

## How to debug locally

```powershell
cd unmask-backend
$env:UNMASK_DEBUG="1"
$env:DEBUG_SCORES="1"
.\venv\Scripts\python.exe -m uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

```powershell
cd unmask-backend
.\venv\Scripts\python.exe tests\test_detection_pipeline.py
```

---

## Recommendations

1. **Retrain fairness** periodically as fake dataset grows; keep val split fixed for comparison.
2. **Log fusion mode** in production when fairness correction triggers (audit bias).
3. **Do not commit `venv/`** — already in `.gitignore`.
4. **Pin dependency versions** in `requirements.txt` for reproducibility.
5. **Future:** Per-demographic calibration or learned ensemble weights instead of hand-tuned fusion.

---

## Files changed in this audit

- `unmask-backend/model.py` — fusion, scoring, logging, thresholds
- `unmask-backend/app.py` — `prob_fake` in response
- `unmask-backend/diagnostics.py` — new
- `unmask-backend/tests/test_detection_pipeline.py` — new
- `mobile/services/api.js` — `probFake` field
- `mobile/App.js` — display semantics
- `docs/DEBUGGING_REPORT.md` — this report
- `docs/DEPENDENCY_REPORT.md` — version report
