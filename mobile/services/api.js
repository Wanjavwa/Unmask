/**
 * Backend API for image verification.
 * Set API_BASE_URL to your machine's local IP when testing on a physical device.
 * Android emulator: use http://10.0.2.2:8000
 * iOS simulator: use http://localhost:8000
 */

import { Platform } from "react-native";

// For Android emulator: use 10.0.2.2
// For Android physical device: use your computer's LAN IP 
// For iOS simulator/web: use localhost
// Update ANDROID_PHYSICAL_IP if your computer's IP changes
// Run "ipconfig" on your PC and use the IPv4 Address (Wi-Fi or Ethernet). Update when your IP changes.
const ANDROID_PHYSICAL_IP = "100.110.186.218";

const BACKEND_PORT = 8011;

function num(v) {
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

/** Build metrics from debug_scores + developer_report when analysis_details is absent. */
function parseMetricsFromLegacy(json) {
  const dbg = json.debug_scores;
  const dev = json.developer_report;
  if (!dbg || typeof dbg !== "object") return null;

  const entropy = (dev && dev.entropy) || {};
  const disagreement = (dev && dev.disagreement) || {};

  return {
    policyVersion:
      typeof dbg.evaluation_policy_version === "string"
        ? dbg.evaluation_policy_version
        : typeof json.evaluation_policy_version === "string"
          ? json.evaluation_policy_version
          : null,
    models: {
      efficientnetB4: num(dbg.effb4),
      xception: num(dbg.xception),
      fairness: num(dbg.fairness),
    },
    entropy: {
      efficientnetB4: num(entropy.effb4),
      xception: num(entropy.xception),
      fairness: num(entropy.fairness),
      average: num(entropy.avg_entropy),
    },
    ensemble: {
      weightedMean: num(dbg.weighted_before_fusion),
      calibratedProbFake: num(dbg.prob_fake ?? dbg.ensemble ?? json.prob_fake),
      probFakeMain: num(dbg.prob_fake_main_detectors ?? json.prob_fake_main_detectors),
      classificationProb: num(dbg.classification_prob),
      fusionMode: typeof dbg.fusion_mode === "string" ? dbg.fusion_mode : null,
    },
    disagreement: {
      spread: num(dbg.main_spread ?? disagreement.spread),
      tier:
        typeof dbg.disagreement_tier === "string"
          ? dbg.disagreement_tier
          : typeof disagreement.tier === "string"
            ? disagreement.tier
            : null,
      agreementScore: num(dbg.agreement_score ?? disagreement.agreement_score ?? json.agreement_score),
      mainDirectionConsistent:
        dbg.main_direction_consistent === true
          ? true
          : dbg.main_direction_consistent === false
            ? false
            : null,
      mainAvg: num(disagreement.main_avg),
    },
    decision: {
      reason: typeof dbg.decision_reason === "string" ? dbg.decision_reason : null,
      labelThresholdOnly: null,
    },
    weights: dbg.weights && typeof dbg.weights === "object" ? dbg.weights : null,
  };
}

function parseMetricsFromTopLevel(json) {
  const ms = json.model_scores;
  if (!ms || typeof ms !== "object") return null;
  const ent = json.entropy_scores || {};
  return {
    policyVersion:
      typeof json.evaluation_policy_version === "string"
        ? json.evaluation_policy_version
        : null,
    models: {
      efficientnetB4: num(ms.efficientnet_b4),
      xception: num(ms.xception),
      fairness: num(ms.fairness),
    },
    entropy: {
      efficientnetB4: num(ent.efficientnet_b4),
      xception: num(ent.xception),
      fairness: num(ent.fairness),
      average: num(ent.average),
    },
    ensemble: {
      weightedMean: num(json.weighted_mean),
      calibratedProbFake: num(json.prob_fake),
      probFakeMain: num(json.prob_fake_main_detectors),
      classificationProb: num(json.classification_prob),
      fusionMode: typeof json.fusion_mode === "string" ? json.fusion_mode : null,
    },
    disagreement: {
      spread: num(json.main_spread),
      tier: typeof json.disagreement_tier === "string" ? json.disagreement_tier : null,
      agreementScore: num(json.agreement_score),
      mainDirectionConsistent: null,
      mainAvg: null,
    },
    decision: {
      reason: typeof json.decision_reason === "string" ? json.decision_reason : null,
      labelThresholdOnly: null,
    },
    weights: null,
  };
}

function hasAnyModelScores(metrics) {
  if (!metrics || !metrics.models) return false;
  const { efficientnetB4, xception, fairness } = metrics.models;
  return [efficientnetB4, xception, fairness].some((v) => v != null);
}

/** Normalize backend analysis_details for the results UI. */
function parseAnalysisDetails(raw) {
  if (!raw || typeof raw !== "object") return null;

  const models = raw.models || {};
  const entropy = raw.entropy || {};
  const ensemble = raw.ensemble || {};
  const disagreement = raw.disagreement || {};
  const decision = raw.decision || {};

  return {
    policyVersion:
      typeof raw.policy_version === "string" ? raw.policy_version : null,
    models: {
      efficientnetB4: num(models.efficientnet_b4),
      xception: num(models.xception),
      fairness: num(models.fairness),
    },
    entropy: {
      efficientnetB4: num(entropy.efficientnet_b4),
      xception: num(entropy.xception),
      fairness: num(entropy.fairness),
      average: num(entropy.average),
    },
    ensemble: {
      weightedMean: num(ensemble.weighted_mean),
      calibratedProbFake: num(ensemble.calibrated_prob_fake),
      probFakeMain: num(ensemble.prob_fake_main_detectors),
      classificationProb: num(ensemble.classification_prob),
      fusionMode:
        typeof ensemble.fusion_mode === "string" ? ensemble.fusion_mode : null,
    },
    disagreement: {
      spread: num(disagreement.spread),
      tier: typeof disagreement.tier === "string" ? disagreement.tier : null,
      agreementScore: num(disagreement.agreement_score),
      mainDirectionConsistent:
        disagreement.main_direction_consistent === true
          ? true
          : disagreement.main_direction_consistent === false
            ? false
            : null,
      mainAvg: num(disagreement.main_avg),
    },
    decision: {
      reason: typeof decision.reason === "string" ? decision.reason : null,
      labelThresholdOnly:
        typeof decision.label_threshold_only === "string"
          ? decision.label_threshold_only
          : null,
    },
    weights: raw.weights && typeof raw.weights === "object" ? raw.weights : null,
  };
}

const API_BASE_URL =
  __DEV__ && Platform.OS === "android"
    ? `http://${ANDROID_PHYSICAL_IP}:${BACKEND_PORT}`
    : __DEV__ && (Platform.OS === "ios" || Platform.OS === "web")
      ? `http://localhost:${BACKEND_PORT}`
      : `http://YOUR_LOCAL_IP:${BACKEND_PORT}`;

/**
 * Upload image to backend and return detection result.
 * @param {string} uri - Local file URI (file:// or content://)
 * @returns {Promise<{ label: string, confidence: number, explanation: string, disclaimer?: string }>}
 * @throws {Error} When request fails or response is invalid
 */
export async function scanImage(uri) {
  const data = new FormData();

  if (Platform.OS === "web") {
    const response = await fetch(uri);
    const blob = await response.blob();
    const file = new File([blob], "upload.jpg", { type: "image/jpeg" });
    data.append("file", file);
  } else {
    data.append("file", {
      uri,
      name: "upload.jpg",
      type: "image/jpeg",
    });
  }

  const res = await fetch(`${API_BASE_URL}/detect-image`, {
    method: "POST",
    body: data,
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(
      res.status === 422 || res.status === 400
        ? "Invalid image. Please choose a valid photo."
        : res.status >= 500
          ? "Server error. Please try again later."
          : text || `Request failed (${res.status})`
    );
  }

  const json = await res.json();
  if (__DEV__) {
    console.log("Backend response:", json);
    if (json.evaluation_policy_version && json.evaluation_policy_version !== "3.3") {
      console.warn(
        "Unmask backend may be outdated. Expected evaluation_policy_version 3.3, got:",
        json.evaluation_policy_version
      );
    }
    if (!json.evaluation_policy_version) {
      console.warn(
        "Unmask backend is OUTDATED (missing evaluation_policy_version). Restart unmask-backend."
      );
    }
  }

  if (json == null || typeof json !== "object") {
    throw new Error("Invalid response from server.");
  }

  const label = json.label;
  const overallConfidence = Number(json.confidence);
  const probFake =
    json.prob_fake != null ? Number(json.prob_fake) : overallConfidence;
  const confidenceLevel =
    typeof json.confidence_level === "string" ? json.confidence_level : "Inconclusive";
  const analysisReliable = json.analysis_reliable === true;
  const agreementScore =
    json.agreement_score != null ? Number(json.agreement_score) : null;
  const probFakeMain =
    json.prob_fake_main_detectors != null ? Number(json.prob_fake_main_detectors) : null;
  const verdictStrength =
    json.verdict_strength != null ? Number(json.verdict_strength) : null;
  const explanation = json.explanation;
  const disclaimer = json.disclaimer;
  const debugScores = json.debug_scores;
  const developerReport = json.developer_report;
  let analysisDetails = parseAnalysisDetails(json.analysis_details);
  if (!hasAnyModelScores(analysisDetails)) {
    analysisDetails =
      parseMetricsFromTopLevel(json) ||
      parseMetricsFromLegacy(json) ||
      analysisDetails;
  }
  if (!hasAnyModelScores(analysisDetails)) {
    console.warn(
      "Unmask: no model scores in API response. Restart backend (port",
      BACKEND_PORT + ")",
      "to get analysis_details."
    );
  }

  if (typeof label !== "string" || !Number.isFinite(overallConfidence)) {
    throw new Error("Invalid response format from server.");
  }

  return {
    label,
    /** Overall confidence in the assessment (agreement, face quality, fusion). */
    confidence: Math.max(0, Math.min(1, overallConfidence)),
    confidenceLevel,
    analysisReliable,
    agreementScore,
    probFakeMain,
    /** How decisive the fake probability is (distance from 50%). */
    verdictStrength:
      verdictStrength != null && Number.isFinite(verdictStrength)
        ? Math.max(0, Math.min(1, verdictStrength))
        : null,
    probFake: Math.max(0, Math.min(1, Number.isFinite(probFake) ? probFake : overallConfidence)),
    explanation: typeof explanation === "string" ? explanation : "",
    disclaimer: typeof disclaimer === "string" ? disclaimer : undefined,
    analysisDetails,
    debugScores: debugScores && typeof debugScores === "object" ? debugScores : undefined,
    developerReport:
      developerReport && typeof developerReport === "object" ? developerReport : undefined,
  };
}
