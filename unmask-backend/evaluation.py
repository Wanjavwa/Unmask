"""
================================================================================
EVALUATION.PY — Policy engine (labels, ensemble math, explanations)
================================================================================

ROLE IN THE SYSTEM
------------------
Receives three model probabilities + face metadata from model.py and returns:
  - Final user-facing label (Likely authentic, Likely deepfake, Mixed signals, …)
  - Calibrated prob_fake (shown in UI bar)
  - overall_confidence + confidence_level string
  - user_explanation, disclaimer, developer_report

PIPELINE POSITION
-----------------
  model.py (raw probs)  →  evaluation.build_evaluation()  →  app.py (JSON)

Policy v3.3 summary:
  spread = |EffB4 − Xception| only (fairness excluded).
  spread > 0.25 + models straddle 0.5 → Mixed signals.
  Otherwise → calibrated weighted prob_fake:
      ≤ 35% Likely authentic | ≥ 65% Likely deepfake | else Inconclusive.
  Confidence tiers (0.15 / 0.30) affect overall_confidence display only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

# ==============================================================================
# POLICY CONSTANTS — thresholds, weights, spread gates
# ==============================================================================
# These numbers define v3.3 behavior. Changing them changes labels without
# retraining any neural network.
# ==============================================================================
WEIGHT_EFFB4 = 0.40
WEIGHT_XCEPTION = 0.40
WEIGHT_FAIRNESS = 0.20

THRESHOLD_FAKE = 0.65
THRESHOLD_REAL = 0.35

# Main-detector spread tiers (|effb4 − xception|) — confidence / fusion
SPREAD_LOW = 0.15
SPREAD_HIGH = 0.30

# Mixed signals: only when spread > this AND models straddle 0.5
SPREAD_MIXED_GATE = 0.25

MIN_OVERALL_FOR_STRONG_LABEL = 0.55
MIN_OVERALL_FOR_STRONG_MODERATE = 0.45  # lower bar when moderate + directional OK
MIN_OVERALL_NOT_UNRELIABLE = 0.45
MAIN_DIRECTION_THRESHOLD = 0.5  # used only in MODERATE tier

EVALUATION_POLICY_VERSION = "3.3"

MAX_BERNoulli_ENTROPY = math.log(2.0)

LABEL_LIKELY_AUTHENTIC = "Likely authentic"
LABEL_LIKELY_DEEPFAKE = "Likely deepfake"
LABEL_MIXED_SIGNALS = "Mixed signals"
LABEL_INCONCLUSIVE = "Inconclusive"
LABEL_UNABLE = "Unable to analyze reliably"
LABEL_INSUFFICIENT_FACE = "Insufficient facial evidence"
LABEL_NOVEL = "Possibly AI-generated (novel pattern)"

STRONG_LABELS = frozenset({LABEL_LIKELY_AUTHENTIC, LABEL_LIKELY_DEEPFAKE})
GATED_LABELS = frozenset({
    LABEL_MIXED_SIGNALS,
    LABEL_INCONCLUSIVE,
    LABEL_UNABLE,
    LABEL_INSUFFICIENT_FACE,
    LABEL_NOVEL,
})


# ==============================================================================
# DATA CONTRACTS — what flows between model.py and this module
# ==============================================================================
# ModelOutputs: softmax fake-class probabilities from each network.
# PreprocessInfo: whether OpenCV found a face (affects labels + confidence).
# DisagreementMetrics / EnsembleResult: internal spread + fusion state.
# EvaluationResult: final bundle returned to model.py → app.py.
# ==============================================================================
@dataclass
class ModelOutputs:
    prob_effb4: float
    prob_xception: float
    prob_fairness: float
    logits_effb4: list[float] | None = None
    logits_xception: list[float] | None = None
    logits_fairness: list[float] | None = None


@dataclass
class PreprocessInfo:
    face_detected: bool
    face_note: str
    face_bbox: dict[str, int] | None = None
    face_size: tuple[int, int] | None = None


@dataclass
class DisagreementMetrics:
    """Disagreement from main detectors only (EffB4 vs Xception)."""
    spread: float  # abs(effb4 - xception)
    main_avg: float  # (effb4 + xception) / 2
    agreement_score: float  # max(0, 1 - spread/0.5)
    tier: str  # "low" | "moderate" | "high"
    high_disagreement: bool
    moderate_disagreement: bool
    low_disagreement: bool
    # Directional consensus (both >= 0.5 or both < 0.5) — evaluated always, used only in MODERATE tier
    main_direction_consistent: bool
    fairness_prob: float
    fairness_delta_from_main: float  # fairness - main_avg


@dataclass
class EnsembleResult:
    weighted: float
    raw_after_fusion: float
    raw_display: float
    prob_fake: float
    prob_fake_main: float
    fusion_mode: str
    fusion_details: dict[str, float]
    disagreement: DisagreementMetrics


@dataclass
class ReliabilityAssessment:
    face_detected: bool
    spread: float
    agreement_score: float
    disagreement_tier: str
    overall_confidence: float
    avg_entropy: float
    fusion_mode: str
    issues: list[str]
    can_issue_strong_label: bool


@dataclass
class DecisionResult:
    label: str
    reason_code: str
    tentative_lean: str


@dataclass
class EvaluationResult:
    label: str
    prob_fake: float
    overall_confidence: float
    confidence_level: str
    verdict_strength: float
    analysis_reliable: bool
    agreement_score: float
    user_explanation: str
    disclaimer: str
    developer_report: dict[str, Any] = field(default_factory=dict)


# ==============================================================================
# SMALL MATH HELPERS — calibration, entropy, threshold comparisons
# ==============================================================================
# calibrate_probability: maps raw [0,1] to display range (linear stretch).
# bernoulli_entropy: uncertainty of a single model's fake probability.
# _prob_indicates_*: label thresholds with 4-decimal rounding for stability.
# ==============================================================================
def calibrate_probability(p: float) -> float:
    return float(min(max((p - 0.05) / 0.9, 0.01), 0.99))


def bernoulli_entropy(p: float, eps: float = 1e-6) -> float:
    p = float(min(max(p, eps), 1.0 - eps))
    return float(-(p * math.log(p) + (1.0 - p) * math.log(1.0 - p)))


def verdict_strength(prob_fake: float) -> float:
    return float(min(max(abs(prob_fake - 0.5) * 2.0, 0.0), 1.0))


def _prob_indicates_fake(p: float) -> bool:
    return round(p, 4) >= THRESHOLD_FAKE


def _prob_indicates_real(p: float) -> bool:
    return round(p, 4) <= THRESHOLD_REAL


def _prob_in_ambiguous_zone(p: float) -> bool:
    return THRESHOLD_REAL < round(p, 4) < THRESHOLD_FAKE


def spread_triggers_mixed_gate(spread: float) -> bool:
    """True when spread is strictly greater than SPREAD_MIXED_GATE (0.25)."""
    return spread > SPREAD_MIXED_GATE


def main_detectors_same_direction(prob_effb4: float, prob_xception: float) -> bool:
    """Both main models on the same side of 50% fake (both ≥ 0.5 or both < 0.5)."""
    both_fake = prob_effb4 >= MAIN_DIRECTION_THRESHOLD and prob_xception >= MAIN_DIRECTION_THRESHOLD
    both_real = prob_effb4 < MAIN_DIRECTION_THRESHOLD and prob_xception < MAIN_DIRECTION_THRESHOLD
    return both_fake or both_real


def weighted_ensemble(prob_effb4: float, prob_xception: float, prob_fairness: float) -> float:
    return (
        WEIGHT_EFFB4 * prob_effb4
        + WEIGHT_XCEPTION * prob_xception
        + WEIGHT_FAIRNESS * prob_fairness
    )


# ==============================================================================
# MAIN-DETECTOR DISAGREEMENT — spread tier and directional consistency
# ==============================================================================
# compute_disagreement(): spread = |EffB4 - Xception|; fairness only logged.
# spread_triggers_mixed_gate / main_detectors_same_direction: Mixed signals gate.
# Used for: Mixed signals gate, confidence factors, UI breakdown.
# ==============================================================================
def compute_disagreement(
    prob_effb4: float, prob_xception: float, prob_fairness: float
) -> DisagreementMetrics:
    """
    Spread uses ONLY EffB4 and Xception.
    Fairness is tracked for logging / light confidence adjustment only.
    """
    spread = abs(prob_effb4 - prob_xception)
    spread = round(spread, 4)
    main_avg = 0.5 * (prob_effb4 + prob_xception)
    agreement_score = float(max(0.0, 1.0 - spread / 0.5))

    if spread >= SPREAD_HIGH:
        tier = "high"
        high, moderate, low = True, False, False
    elif spread >= SPREAD_LOW:
        tier = "moderate"
        high, moderate, low = False, True, False
    else:
        tier = "low"
        high, moderate, low = False, False, True

    direction_ok = main_detectors_same_direction(prob_effb4, prob_xception)

    return DisagreementMetrics(
        spread=spread,
        main_avg=main_avg,
        agreement_score=agreement_score,
        tier=tier,
        high_disagreement=high,
        moderate_disagreement=moderate,
        low_disagreement=low,
        main_direction_consistent=direction_ok,
        fairness_prob=prob_fairness,
        fairness_delta_from_main=prob_fairness - main_avg,
    )


# ==============================================================================
# ENSEMBLE FUSION — weighted mean + optional fairness nudge
# ==============================================================================
# fuse_ensemble(): 40/40/20 blend; 10% fairness nudge only when spread < 0.15.
# apply_disagreement_to_display_probability(): shrink toward 50% if spread ≥ 0.15.
# classification_probability(): returns calibrated prob_fake for label thresholds.
# build_ensemble(): orchestrates the above → EnsembleResult.
# ==============================================================================
def fuse_ensemble(
    prob_effb4: float,
    prob_xception: float,
    prob_fairness: float,
    disagreement: DisagreementMetrics,
) -> tuple[float, str, dict[str, float]]:
    """
    Default: weighted mean of all three models.
    Fairness does NOT pull the score when main detectors disagree (spread ≥ 0.15).
    When main detectors agree (low spread), apply at most a 10% nudge toward fairness.
    """
    weighted = weighted_ensemble(prob_effb4, prob_xception, prob_fairness)
    details: dict[str, float] = {
        "main_avg": round(disagreement.main_avg, 4),
        "main_spread": round(disagreement.spread, 4),
        "weighted": round(weighted, 4),
        "fairness": round(prob_fairness, 4),
    }

    if disagreement.high_disagreement or disagreement.moderate_disagreement:
        return weighted, "none_main_disagreement", {
            **details,
            "note": "fairness nudge disabled when main spread >= 0.15",
        }

    # Low spread only: light 10% nudge (fairness has reduced authority)
    if prob_fairness < 0.25 and disagreement.main_avg > 0.55:
        nudged = 0.90 * weighted + 0.10 * prob_fairness
        return nudged, "fairness_nudge_light_real", {**details, "blend": "90% weighted + 10% fairness"}

    if prob_fairness > 0.75 and disagreement.main_avg < 0.45:
        nudged = 0.90 * weighted + 0.10 * prob_fairness
        return nudged, "fairness_nudge_light_fake", {**details, "blend": "90% weighted + 10% fairness"}

    return weighted, "none", details


def apply_disagreement_to_display_probability(
    raw: float, disagreement: DisagreementMetrics
) -> float:
    """
    Shrink toward 0.5 only when spread is MODERATE or HIGH.
    LOW spread: models already agree — keep the fused score unchanged.
    """
    if disagreement.low_disagreement:
        return float(raw)
    a = disagreement.agreement_score
    return float(0.5 + (raw - 0.5) * a)


def classification_probability(ensemble: EnsembleResult) -> float:
    """
    Probability used for label thresholds (0.35 / 0.65) after face + spread gates.

    Always the calibrated weighted ensemble (0.4 EffB4 + 0.4 Xception + 0.2 Fairness),
    matching the display fake probability — not the main-detector-only average.
    """
    return ensemble.prob_fake


def build_ensemble(models: ModelOutputs) -> EnsembleResult:
    disagreement = compute_disagreement(
        models.prob_effb4, models.prob_xception, models.prob_fairness
    )
    weighted = weighted_ensemble(
        models.prob_effb4, models.prob_xception, models.prob_fairness
    )
    raw_fusion, fusion_mode, fusion_details = fuse_ensemble(
        models.prob_effb4, models.prob_xception, models.prob_fairness, disagreement
    )
    raw_display = apply_disagreement_to_display_probability(raw_fusion, disagreement)
    prob_fake = calibrate_probability(raw_display)
    prob_fake_main = calibrate_probability(disagreement.main_avg)

    return EnsembleResult(
        weighted=weighted,
        raw_after_fusion=raw_fusion,
        raw_display=raw_display,
        prob_fake=prob_fake,
        prob_fake_main=prob_fake_main,
        fusion_mode=fusion_mode,
        fusion_details=fusion_details,
        disagreement=disagreement,
    )


# ==============================================================================
# OVERALL CONFIDENCE — reliability score for UI "Confidence: High/Low/…"
# ==============================================================================
# Multiplicative factors from spread, face, entropy, disagreement tier, fairness gap.
# v3.3: does NOT block Likely authentic/deepfake — only affects confidence_level().
# ==============================================================================
def overall_confidence(
    disagreement: DisagreementMetrics,
    face_detected: bool,
    fusion_mode: str,
    avg_entropy: float,
) -> float:
    """Reliability from main-detector agreement, face quality, entropy, light fairness delta."""
    agreement = disagreement.agreement_score
    face_factor = 1.0 if face_detected else 0.65
    entropy_norm = min(avg_entropy / MAX_BERNoulli_ENTROPY, 1.0)
    entropy_factor = 1.0 - 0.20 * entropy_norm

    if disagreement.high_disagreement:
        disagreement_factor = 0.40
    elif disagreement.moderate_disagreement:
        disagreement_factor = 0.70
    else:
        disagreement_factor = 1.0

    # Fairness far from main avg → small confidence reduction (does not change label)
    fairness_gap = abs(disagreement.fairness_delta_from_main)
    fairness_factor = 1.0 - 0.08 * min(fairness_gap / 0.5, 1.0)

    fusion_factor = 0.97 if fusion_mode.startswith("fairness_nudge") else 1.0

    raw = agreement * face_factor * entropy_factor * disagreement_factor * fairness_factor * fusion_factor
    return float(min(max(raw, 0.0), 1.0)    )


# ==============================================================================
# RELIABILITY ASSESSMENT — logging + can_issue_strong_label flag
# ==============================================================================
# Records issues (no face, spread above gate, etc.). can_strong is true when
# face present and not mixed_blocked — used in developer_report only in v3.3.
# ==============================================================================
def assess_reliability(
    preprocess: PreprocessInfo,
    disagreement: DisagreementMetrics,
    overall: float,
    avg_entropy: float,
    fusion_mode: str,
    class_prob: float,
) -> ReliabilityAssessment:
    """
    Labels use calibrated prob_fake thresholds only (after face + Mixed gate).
    can_issue_strong_label is true whenever that threshold path applies (for reporting).
    """
    issues: list[str] = []

    if not preprocess.face_detected:
        issues.append("no_face_detected")
    if spread_triggers_mixed_gate(disagreement.spread):
        issues.append("spread_above_mixed_gate")
        if not disagreement.main_direction_consistent:
            issues.append("direction_conflict_across_half")
    elif disagreement.moderate_disagreement:
        issues.append("moderate_main_detector_disagreement")
    if abs(disagreement.fairness_delta_from_main) > 0.35:
        issues.append("fairness_diverges_from_main")  # logged only

    mixed_blocked = spread_triggers_mixed_gate(
        disagreement.spread
    ) and not disagreement.main_direction_consistent

    can_strong = preprocess.face_detected and not mixed_blocked

    return ReliabilityAssessment(
        face_detected=preprocess.face_detected,
        spread=disagreement.spread,
        agreement_score=disagreement.agreement_score,
        disagreement_tier=disagreement.tier,
        overall_confidence=overall,
        avg_entropy=avg_entropy,
        fusion_mode=fusion_mode,
        issues=issues,
        can_issue_strong_label=can_strong,
    )


def _tentative_lean(prob_fake: float) -> str:
    if prob_fake >= 0.55:
        return "fake"
    if prob_fake <= 0.45:
        return "real"
    return "neutral"


# ==============================================================================
# LABEL DECISION — face → Mixed gate → calibrated prob thresholds
# ==============================================================================
# decide_label(): the authoritative verdict logic for v3.3.
# Order: no face → Mixed (spread>0.25 & straddle 0.5) → novelty → 65%/35% → Inconclusive.
# confidence_level(): maps overall_confidence to Very High / High / … for UI.
# ==============================================================================
def decide_label(
    prob_fake: float,
    class_prob: float,
    novelty: bool,
    reliability: ReliabilityAssessment,
    disagreement: DisagreementMetrics,
) -> DecisionResult:
    """
    Face -> Mixed gate (spread > 0.25 + straddle 0.5) -> else calibrated prob thresholds only.
    Low spread and same-side high spread use the same 35% / 65% rules on class_prob.
    """
    lean = _tentative_lean(prob_fake)

    if not reliability.face_detected:
        return DecisionResult(LABEL_INSUFFICIENT_FACE, "no_face_detected", lean)

    if spread_triggers_mixed_gate(disagreement.spread) and not disagreement.main_direction_consistent:
        return DecisionResult(LABEL_MIXED_SIGNALS, "spread_direction_conflict", lean)

    if novelty:
        return DecisionResult(LABEL_NOVEL, "novelty_pattern", lean)
    if _prob_indicates_fake(class_prob):
        return DecisionResult(LABEL_LIKELY_DEEPFAKE, "threshold_fake", lean)
    if _prob_indicates_real(class_prob):
        return DecisionResult(LABEL_LIKELY_AUTHENTIC, "threshold_real", lean)
    return DecisionResult(LABEL_INCONCLUSIVE, "ambiguous_probability", lean)


def confidence_level(overall: float, label: str) -> str:
    if label in GATED_LABELS or label == LABEL_INCONCLUSIVE:
        return "Inconclusive"
    if overall >= 0.82:
        return "Very High"
    if overall >= 0.62:
        return "High"
    if overall >= 0.40:
        return "Moderate"
    if overall >= 0.18:
        return "Low"
    return "Inconclusive"


# ==============================================================================
# USER-FACING TEXT — plain-language explanations per label
# ==============================================================================
# build_user_explanation(): one paragraph shown under the result badge.
# Uses label + prob_fake + spread; avoids raw debug tokens in user text.
# ==============================================================================
def build_user_explanation(
    label: str,
    prob_fake: float,
    ensemble: EnsembleResult,
    confidence_level_str: str,
    preprocess: PreprocessInfo,
    models: ModelOutputs,
    decision: DecisionResult,
) -> str:
    d = ensemble.disagreement

    if label == LABEL_INSUFFICIENT_FACE:
        return (
            "Insufficient facial evidence was detected to make a reliable determination. "
            "A center crop was analyzed because no clear face was found. "
            "Please use a well-lit photo with a visible face for a stronger result."
        )

    if label == LABEL_UNABLE:
        return (
            "Unable to analyze this image reliably. "
            "Overall confidence was too low to support a clear verdict."
        )

    if label == LABEL_MIXED_SIGNALS:
        extra = ""
        if spread_triggers_mixed_gate(d.spread) and not d.main_direction_consistent:
            extra = " Main detectors pointed in different directions (one favored real, one favored fake)."
        return (
            f"Mixed signals detected. EfficientNet-B4 and XceptionNet disagreed "
            f"(spread {_pct(d.spread)}). No firm authentic vs. deepfake verdict is given."
            + extra
        )

    if label == LABEL_INCONCLUSIVE:
        return (
            f"Inconclusive. Estimated fake probability is {_pct(prob_fake)}. "
            f"Main detector spread was {_pct(d.spread)} ({d.tier} disagreement)."
        )

    if label == LABEL_NOVEL:
        return "Possibly AI-generated using an unfamiliar pattern. Treat as indicative, not definitive."

    if label == LABEL_LIKELY_DEEPFAKE:
        return (
            f"Analysis suggests this image is likely AI-generated or manipulated "
            f"(fake probability: {_pct(prob_fake)}). Main detectors agreed (spread {_pct(d.spread)}). "
            f"Confidence: {confidence_level_str}."
        )

    if label == LABEL_LIKELY_AUTHENTIC:
        return (
            f"Analysis suggests this image is likely authentic "
            f"(fake probability: {_pct(prob_fake)}). Main detectors agreed (spread {_pct(d.spread)}). "
            f"Confidence: {confidence_level_str}."
        )

    return "Analysis complete."


def _pct(p: float) -> str:
    return f"{round(p * 100)}%"


# ==============================================================================
# DEVELOPER REPORT — full structured audit trail (optional in API)
# ==============================================================================
# build_developer_report(): pipeline steps, disagreement block, ensemble block,
# decision_policy block. app.py flattens parts into analysis_details for mobile.
# ==============================================================================
def build_developer_report(
    models: ModelOutputs,
    preprocess: PreprocessInfo,
    ensemble: EnsembleResult,
    entropies: dict[str, float],
    avg_entropy: float,
    overall: float,
    level: str,
    decision: DecisionResult,
    reliability: ReliabilityAssessment,
    novelty: bool,
    label_threshold_only: str,
) -> dict[str, Any]:
    d = ensemble.disagreement
    return {
        "policy_version": EVALUATION_POLICY_VERSION,
        "pipeline": [
            "softmax_per_model",
            "main_detector_spread",
            "weighted_ensemble",
            "optional_fairness_light_nudge",
            "disagreement_shrink",
            "calibrate",
            "reliability_gated_label",
        ],
        "disagreement": {
            "spread_formula": "abs(effb4 - xception)",
            "spread": round(d.spread, 4),
            "tier": d.tier,
            "thresholds": {
                "low": f"spread < {SPREAD_LOW}",
                "moderate": f"{SPREAD_LOW} ≤ spread < {SPREAD_HIGH}",
                "high": f"spread ≥ {SPREAD_HIGH}",
            },
            "agreement_score": round(d.agreement_score, 4),
            "agreement_formula": "max(0, 1 - spread / 0.5)",
            "main_avg": round(d.main_avg, 4),
            "fairness_prob": round(d.fairness_prob, 4),
            "fairness_delta_from_main": round(d.fairness_delta_from_main, 4),
            "note": "Fairness is NOT included in spread tier or Mixed signals triggers.",
        },
        "models": {
            "efficientnet_b4_fake_prob": round(models.prob_effb4, 4),
            "xception_fake_prob": round(models.prob_xception, 4),
            "fairness_fake_prob": round(models.prob_fairness, 4),
            "logits": {
                "effb4": models.logits_effb4,
                "xception": models.logits_xception,
                "fairness": models.logits_fairness,
            },
        },
        "ensemble": {
            "weights": {"effb4": WEIGHT_EFFB4, "xception": WEIGHT_XCEPTION, "fairness": WEIGHT_FAIRNESS},
            "weighted_mean": round(ensemble.weighted, 4),
            "fusion_mode": ensemble.fusion_mode,
            "fusion_details": ensemble.fusion_details,
            "calibrated_prob_fake": round(ensemble.prob_fake, 4),
            "prob_fake_main_detectors": round(ensemble.prob_fake_main, 4),
        },
        "decision_policy": {
            "label_threshold_only": label_threshold_only,
            "final_label": decision.label,
            "reason_code": decision.reason_code,
            "reliability_issues": reliability.issues,
            "can_issue_strong_label": reliability.can_issue_strong_label,
            "classification_probability_source": "calibrated_weighted_ensemble_prob_fake",
            "mixed_signals_gate": {
                "when": f"spread > {SPREAD_MIXED_GATE}",
                "rule": "both_effb4_and_xception_above_0.5 OR both_below_0.5 else Mixed signals",
                "main_direction_consistent": d.main_direction_consistent,
            },
            "label_from_calibrated_prob_only": {
                "after_gates": "prob_fake <= 0.35 | >= 0.65 | else Inconclusive",
                "note": "No separate can_strong confidence gate on label (v3.3)",
            },
            "confidence_tiers": {
                "low": f"spread < {SPREAD_LOW}",
                "moderate": f"{SPREAD_LOW} ≤ spread < {SPREAD_HIGH}",
                "high": f"spread ≥ {SPREAD_HIGH}",
            },
        },
        "entropy": {**{k: round(v, 4) for k, v in entropies.items()}, "avg_entropy": round(avg_entropy, 4)},
        "confidence": {"overall_confidence": round(overall, 4), "confidence_level": level},
        "developer_summary_lines": [
            f"EfficientNet-B4: {_pct(models.prob_effb4)} fake",
            f"XceptionNet: {_pct(models.prob_xception)} fake",
            f"Fairness (debug only): {_pct(models.prob_fairness)} fake",
            f"Main spread: {_pct(d.spread)} ({d.tier})",
            f"Weighted → calibrated: {_pct(ensemble.prob_fake)}",
            f"Final: {decision.label} ({decision.reason_code})",
        ],
    }


def _label_from_prob_threshold_only(prob_fake: float, novelty: bool) -> str:
    if novelty:
        return LABEL_NOVEL
    if _prob_indicates_fake(prob_fake):
        return LABEL_LIKELY_DEEPFAKE
    if _prob_indicates_real(prob_fake):
        return LABEL_LIKELY_AUTHENTIC
    return LABEL_INCONCLUSIVE


# ==============================================================================
# ORCHESTRATOR — build_evaluation() ties everything together
# ==============================================================================
# Entry point from model.py. Steps:
#   1. build_ensemble(models)
#   2. entropy per model + overall_confidence
#   3. assess_reliability + decide_label
#   4. build_user_explanation + developer_report + debug_scores
# Returns EvaluationResult consumed by model.predict_deepfake → app.py.
# ==============================================================================
def build_evaluation(models: ModelOutputs, preprocess: PreprocessInfo) -> EvaluationResult:
    ensemble = build_ensemble(models)
    prob_fake = ensemble.prob_fake
    d = ensemble.disagreement

    ent_e = bernoulli_entropy(models.prob_effb4)
    ent_x = bernoulli_entropy(models.prob_xception)
    ent_f = bernoulli_entropy(models.prob_fairness)
    avg_entropy = (ent_e + ent_x + ent_f) / 3.0
    entropies = {"effb4": ent_e, "xception": ent_x, "fairness": ent_f}

    class_prob = classification_probability(ensemble)
    novelty = (class_prob < 0.2) and (avg_entropy < 0.15) and d.low_disagreement
    label_threshold_only = _label_from_prob_threshold_only(class_prob, novelty)

    v_str = verdict_strength(prob_fake)
    overall = overall_confidence(d, preprocess.face_detected, ensemble.fusion_mode, avg_entropy)

    reliability = assess_reliability(
        preprocess, d, overall, avg_entropy, ensemble.fusion_mode, class_prob
    )
    decision = decide_label(prob_fake, class_prob, novelty, reliability, d)
    level = confidence_level(overall, decision.label)
    analysis_reliable = decision.label in STRONG_LABELS

    user_explanation = build_user_explanation(
        decision.label, prob_fake, ensemble, level, preprocess, models, decision
    )

    developer_report = build_developer_report(
        models, preprocess, ensemble, entropies, avg_entropy,
        overall, level, decision, reliability, novelty, label_threshold_only,
    )

    developer_report["debug_scores"] = {
        "evaluation_policy_version": EVALUATION_POLICY_VERSION,
        "effb4": round(models.prob_effb4, 4),
        "xception": round(models.prob_xception, 4),
        "fairness": round(models.prob_fairness, 4),
        "main_spread": round(d.spread, 4),
        "disagreement_tier": d.tier,
        "main_direction_consistent": d.main_direction_consistent,
        "weighted_before_fusion": round(ensemble.weighted, 4),
        "ensemble": round(prob_fake, 4),
        "prob_fake": round(prob_fake, 4),
        "prob_fake_main_detectors": round(ensemble.prob_fake_main, 4),
        "classification_prob": round(class_prob, 4),
        "agreement_score": round(d.agreement_score, 4),
        "overall_confidence": round(overall, 4),
        "confidence_level": level,
        "fusion_mode": ensemble.fusion_mode,
        "decision_reason": decision.reason_code,
        "verdict_strength": round(v_str, 4),
        "analysis_reliable": analysis_reliable,
        "weights": {"effb4": WEIGHT_EFFB4, "xception": WEIGHT_XCEPTION, "fairness": WEIGHT_FAIRNESS},
        "thresholds": {
            "fake": THRESHOLD_FAKE,
            "real": THRESHOLD_REAL,
            "spread_low": SPREAD_LOW,
            "spread_high": SPREAD_HIGH,
            "spread_mixed_gate": SPREAD_MIXED_GATE,
        },
    }

    disclaimer = (
        "Unmask provides probabilistic analysis and may be affected by dataset bias, "
        "lighting, and image quality. It should be used as a decision-support aid, "
        "not as definitive proof."
    )

    return EvaluationResult(
        label=decision.label,
        prob_fake=prob_fake,
        overall_confidence=overall,
        confidence_level=level,
        verdict_strength=v_str,
        analysis_reliable=analysis_reliable,
        agreement_score=d.agreement_score,
        user_explanation=user_explanation,
        disclaimer=disclaimer,
        developer_report=developer_report,
    )
