"""Unit tests for evaluation scoring and explanations."""

from __future__ import annotations

import os
import sys
import unittest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from evaluation import (
    LABEL_INSUFFICIENT_FACE,
    ModelOutputs,
    PreprocessInfo,
    build_evaluation,
    compute_disagreement,
    fuse_ensemble,
    weighted_ensemble,
)


class TestEvaluationMath(unittest.TestCase):
    def test_weighted_ensemble_example(self) -> None:
        # User-reported case: effb4=0.726, xception=0.484, fairness=0.113
        w = weighted_ensemble(0.726, 0.484, 0.113)
        self.assertAlmostEqual(w, 0.5066, places=3)

    def test_fusion_skipped_when_models_disagree(self) -> None:
        d = compute_disagreement(0.726, 0.484, 0.113)
        self.assertEqual(d.tier, "moderate")
        raw, mode, _ = fuse_ensemble(0.726, 0.484, 0.113, d)
        self.assertEqual(mode, "none_main_disagreement")

    def test_fusion_none_when_main_detectors_agree(self) -> None:
        d = compute_disagreement(0.55, 0.54, 0.52)
        self.assertEqual(d.tier, "low")
        raw, mode, _ = fuse_ensemble(0.55, 0.54, 0.52, d)
        self.assertIn(mode, ("none", "fairness_nudge_light_real", "fairness_nudge_light_fake"))

    def test_user_explanation_has_no_raw_debug_tokens(self) -> None:
        ev = build_evaluation(
            ModelOutputs(0.726, 0.484, 0.113),
            PreprocessInfo(False, "No face", None, None),
        )
        self.assertEqual(ev.label, LABEL_INSUFFICIENT_FACE)
        self.assertNotIn("effb4=", ev.user_explanation.lower())
        self.assertNotIn("xception=", ev.user_explanation.lower())
        self.assertNotIn("avg_entropy", ev.user_explanation.lower())

    def test_confidence_level_is_string(self) -> None:
        models = ModelOutputs(0.9, 0.9, 0.9)
        preprocess = PreprocessInfo(True, "face", None, (224, 224))
        ev = build_evaluation(models, preprocess)
        self.assertIn(ev.confidence_level, (
            "Very High", "High", "Moderate", "Low", "Inconclusive"
        ))

    def test_developer_report_has_pipeline(self) -> None:
        models = ModelOutputs(0.5, 0.5, 0.5)
        preprocess = PreprocessInfo(True, "face", None, (100, 100))
        ev = build_evaluation(models, preprocess)
        self.assertIn("pipeline", ev.developer_report)
        self.assertIn("developer_summary_lines", ev.developer_report)


if __name__ == "__main__":
    unittest.main(verbosity=2)
