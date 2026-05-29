from __future__ import annotations

import os
from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from model import predict_deepfake
from evaluation import EVALUATION_POLICY_VERSION

# DEBUG_SCORES=1 → full developer_report in API response
DEBUG_SCORES_ENABLED = os.environ.get("DEBUG_SCORES", "").strip() == "1"


def build_analysis_details(developer_report: dict) -> dict:
    """Structured scores for the client UI (models, entropy, spread, ensemble)."""
    debug = developer_report.get("debug_scores", {})
    entropy = developer_report.get("entropy", {})
    disagreement = developer_report.get("disagreement", {})
    ensemble = developer_report.get("ensemble", {})
    decision = developer_report.get("decision_policy", {})

    return {
        "policy_version": debug.get("evaluation_policy_version"),
        "models": {
            "efficientnet_b4": debug.get("effb4"),
            "xception": debug.get("xception"),
            "fairness": debug.get("fairness"),
        },
        "entropy": {
            "efficientnet_b4": entropy.get("effb4"),
            "xception": entropy.get("xception"),
            "fairness": entropy.get("fairness"),
            "average": entropy.get("avg_entropy"),
        },
        "ensemble": {
            "weighted_mean": debug.get("weighted_before_fusion"),
            "calibrated_prob_fake": debug.get("prob_fake"),
            "prob_fake_main_detectors": debug.get("prob_fake_main_detectors"),
            "classification_prob": debug.get("classification_prob"),
            "fusion_mode": debug.get("fusion_mode") or ensemble.get("fusion_mode"),
        },
        "disagreement": {
            "spread": debug.get("main_spread") or disagreement.get("spread"),
            "tier": debug.get("disagreement_tier") or disagreement.get("tier"),
            "agreement_score": debug.get("agreement_score") or disagreement.get("agreement_score"),
            "main_direction_consistent": debug.get("main_direction_consistent"),
            "main_avg": disagreement.get("main_avg"),
        },
        "decision": {
            "reason": debug.get("decision_reason") or decision.get("reason_code"),
            "label_threshold_only": decision.get("label_threshold_only"),
        },
        "weights": debug.get("weights") or ensemble.get("weights"),
    }


app = FastAPI()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "evaluation_policy_version": EVALUATION_POLICY_VERSION,
    }


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.post("/detect-image")
async def detect_image(file: UploadFile = File(...)):
    try:
        raw = await file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="Empty file.")

        try:
            image = Image.open(BytesIO(raw))
            image.load()
        except UnidentifiedImageError as e:
            raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.") from e

        label, prob_fake, overall_confidence, explanation, disclaimer, developer_report = predict_deepfake(
            image
        )
        debug_scores = developer_report.get("debug_scores", {})
        analysis_details = build_analysis_details(developer_report)

        response: dict = {
            "evaluation_policy_version": debug_scores.get("evaluation_policy_version", "unknown"),
            "label": label,
            "prob_fake": round(float(prob_fake), 4),
            "confidence": round(float(overall_confidence), 4),
            "confidence_level": debug_scores.get("confidence_level", "Inconclusive"),
            "analysis_reliable": bool(debug_scores.get("analysis_reliable", False)),
            "agreement_score": round(float(debug_scores.get("agreement_score", 0)), 4),
            "prob_fake_main_detectors": round(
                float(debug_scores.get("prob_fake_main_detectors", prob_fake)), 4
            ),
            "verdict_strength": round(float(debug_scores.get("verdict_strength", 0)), 4),
            "explanation": explanation,
            "disclaimer": disclaimer,
            "analysis_details": analysis_details,
            "model_scores": {
                "efficientnet_b4": debug_scores.get("effb4"),
                "xception": debug_scores.get("xception"),
                "fairness": debug_scores.get("fairness"),
            },
            "entropy_scores": analysis_details.get("entropy"),
            "main_spread": debug_scores.get("main_spread"),
            "disagreement_tier": debug_scores.get("disagreement_tier"),
            "classification_prob": debug_scores.get("classification_prob"),
            "decision_reason": debug_scores.get("decision_reason"),
            "fusion_mode": debug_scores.get("fusion_mode"),
            "weighted_mean": debug_scores.get("weighted_before_fusion"),
        }
        if DEBUG_SCORES_ENABLED:
            response["debug_scores"] = debug_scores
            response["developer_report"] = developer_report
        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal error during detection: {type(e).__name__}: {e}",
        ) from e
