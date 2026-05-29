"""Spread > 0.25 directional gate; calibrated thresholds (policy v3.3)."""

from __future__ import annotations

import sys
import unittest

_ROOT = __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from evaluation import (
    EVALUATION_POLICY_VERSION,
    LABEL_INCONCLUSIVE,
    LABEL_LIKELY_AUTHENTIC,
    LABEL_LIKELY_DEEPFAKE,
    LABEL_MIXED_SIGNALS,
    STRONG_LABELS,
    ModelOutputs,
    PreprocessInfo,
    build_evaluation,
    compute_disagreement,
    main_detectors_same_direction,
)


class TestTierDecision(unittest.TestCase):
    def test_policy_33(self) -> None:
        self.assertEqual(EVALUATION_POLICY_VERSION, "3.3")

    def test_low_spread_no_directional_gate_72_68(self) -> None:
        d = compute_disagreement(0.72, 0.68, 0.05)
        self.assertTrue(d.low_disagreement)
        ev = build_evaluation(
            ModelOutputs(0.72, 0.68, 0.05),
            PreprocessInfo(True, "face", None, (224, 224)),
        )
        # Label uses calibrated weighted ensemble (~52%), not main-only average
        self.assertEqual(ev.label, LABEL_INCONCLUSIVE)

    def test_low_spread_weighted_authentic_41_33(self) -> None:
        ev = build_evaluation(
            ModelOutputs(0.41, 0.33, 0.05),
            PreprocessInfo(True, "face", None, (224, 224)),
        )
        self.assertEqual(ev.label, LABEL_LIKELY_AUTHENTIC)
        self.assertLess(ev.prob_fake, 0.35)

    def test_low_spread_no_directional_gate_22_28(self) -> None:
        d = compute_disagreement(0.22, 0.28, 0.05)
        self.assertTrue(d.low_disagreement)
        ev = build_evaluation(
            ModelOutputs(0.22, 0.28, 0.05),
            PreprocessInfo(True, "face", None, (224, 224)),
        )
        self.assertEqual(ev.label, LABEL_LIKELY_AUTHENTIC)

    def test_moderate_both_above_half_continues(self) -> None:
        d = compute_disagreement(0.72, 0.55, 0.10)
        self.assertEqual(d.tier, "moderate")
        self.assertTrue(main_detectors_same_direction(0.72, 0.55))
        ev = build_evaluation(
            ModelOutputs(0.72, 0.55, 0.10),
            PreprocessInfo(True, "face", None, (224, 224)),
        )
        self.assertNotEqual(ev.label, LABEL_MIXED_SIGNALS)
        self.assertIn(ev.label, (LABEL_INCONCLUSIVE, LABEL_LIKELY_DEEPFAKE))

    def test_moderate_both_below_half_authentic(self) -> None:
        d = compute_disagreement(0.22, 0.40, 0.10)
        self.assertEqual(d.tier, "moderate")
        self.assertTrue(main_detectors_same_direction(0.22, 0.40))
        ev = build_evaluation(
            ModelOutputs(0.22, 0.40, 0.10),
            PreprocessInfo(True, "face", None, (224, 224)),
        )
        self.assertEqual(ev.label, LABEL_LIKELY_AUTHENTIC)

    def test_moderate_direction_conflict_not_mixed(self) -> None:
        """0.55 vs 0.38 — spread 0.17 ≤ 0.25 → normal labeling, not Mixed."""
        d = compute_disagreement(0.55, 0.38, 0.10)
        self.assertEqual(d.tier, "moderate")
        self.assertFalse(main_detectors_same_direction(0.55, 0.38))
        ev = build_evaluation(
            ModelOutputs(0.55, 0.38, 0.10),
            PreprocessInfo(True, "face", None, (224, 224)),
        )
        self.assertNotEqual(ev.label, LABEL_MIXED_SIGNALS)

    def test_high_spread_direction_conflict_mixed(self) -> None:
        ev = build_evaluation(
            ModelOutputs(0.72, 0.40, 0.10),
            PreprocessInfo(True, "face", None, (224, 224)),
        )
        self.assertEqual(ev.label, LABEL_MIXED_SIGNALS)

    def test_high_spread_same_direction_continues(self) -> None:
        """0.88 vs 0.58 — spread 0.30 but both ≥ 0.5 → not Mixed; weighted labeling."""
        ev = build_evaluation(
            ModelOutputs(0.88, 0.58, 0.10),
            PreprocessInfo(True, "face", None, (224, 224)),
        )
        self.assertNotEqual(ev.label, LABEL_MIXED_SIGNALS)
        self.assertEqual(ev.label, LABEL_INCONCLUSIVE)


if __name__ == "__main__":
    unittest.main(verbosity=2)
