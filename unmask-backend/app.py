from __future__ import annotations

import os
from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from model import predict_deepfake

# If DEBUG_SCORES=1, include debug_scores in the API response
DEBUG_SCORES_ENABLED = os.environ.get("DEBUG_SCORES", "").strip() == "1"

app = FastAPI()

# This allows localhost origins for testing and prevents CRS errors
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8081",
        "http://localhost:8082",
        "http://localhost:19006",
        "http://127.0.0.1:8081",
        "http://127.0.0.1:8082",
        "http://127.0.0.1:19006",
    ],
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

        label, confidence, explanation, disclaimer, debug_scores = predict_deepfake(image)
        response: dict = {
            "label": label,
            "confidence": round(float(confidence), 4),
            "explanation": explanation,
            "disclaimer": disclaimer,
        }
        if DEBUG_SCORES_ENABLED:
            response["debug_scores"] = debug_scores
        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal error during detection: {type(e).__name__}: {e}",
        ) from e
