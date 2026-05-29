# Unmask Evaluation System (policy v3.3)

## Spread (main detectors only)

```
spread = abs(EffB4_fake_prob − Xception_fake_prob)
```

## Flow

```mermaid
flowchart TD
    FACE{Face?} -->|No| NF[Insufficient facial evidence]
    FACE -->|Yes| MIX{spread > 0.25 AND straddle 0.5?}
    MIX -->|Yes| MS[Mixed signals]
    MIX -->|No| CAL[Calibrated weighted prob_fake]
    CAL --> T{prob_fake}
    T -->|≤ 35%| AUTH[Likely authentic]
    T -->|≥ 65%| FAKE[Likely deepfake]
    T -->|else| INC[Inconclusive]
```

**Same calibrated path** for:

- Low spread (models agree)
- Higher spread when both models are on the same side of 50%

## Calibrated score (UI bar + labels)

```
weighted = 0.40×EffB4 + 0.40×Xception + 0.20×Fairness
→ optional fairness nudge (low spread only)
→ shrink toward 50% only if spread ≥ 0.15
→ calibrate: (score − 0.05) / 0.9
= prob_fake
```

## Mixed signals only when

- spread **> 0.25**, and
- one main model **≥ 0.5** and the other **< 0.5**

## overall_confidence

Used for **confidence level** display (Very High / High / …), **not** to block Likely authentic/deepfake (v3.3).

## API

`GET /health` → `evaluation_policy_version: "3.3"`
