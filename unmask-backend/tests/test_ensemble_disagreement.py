"""Main-detector spread (EffB4 vs Xception) — policy v3.0."""

from __future__ import annotations

import os
import sys
import unittest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from evaluation import (
    EVALUATION_POLICY_VERSION,
    LABEL_LIKELY_AUTHENTIC,
    LABEL_LIKELY_DEEPFAKE,
    LABEL_MIXED_SIGNALS,
    STRONG_LABELS,
    ModelOutputs,
    PreprocessInfo,
    build_ensemble,
    build_evaluation,
    compute_disagreement,
    fuse_ensemble,
    weighted_ensemble,
)


class TestMainDetectorSpread(unittest.TestCase):
    def test_policy_version(self) -> None:
        self.assertEqual(EVALUATION_POLICY_VERSION, "3.3")

    def test_spread_is_effb4_minus_xception_only(self) -> None:
        d = compute_disagreement(0.72, 0.40, 0.10)
        self.assertAlmostEqual(d.spread, 0.32, places=3)
        self.assertEqual(d.tier, "high")

    def test_low_spread_72_vs_68(self) -> None:
        d = compute_disagreement(0.72, 0.68, 0.05)
        self.assertAlmostEqual(d.spread, 0.04, places=3)
        self.assertEqual(d.tier, "low")
        self.assertTrue(d.low_disagreement)

    def test_moderate_spread_72_vs_58(self) -> None:
        d = compute_disagreement(0.72, 0.58, 0.10)
        self.assertAlmostEqual(d.spread, 0.14, places=3)
        self.assertEqual(d.tier, "low")  # 0.14 < 0.15

        d2 = compute_disagreement(0.72, 0.55, 0.10)
        self.assertAlmostEqual(d2.spread, 0.17, places=3)
        self.assertEqual(d2.tier, "moderate")

    def test_high_spread_triggers_mixed_with_face(self) -> None:
        ev = build_evaluation(
            ModelOutputs(0.72, 0.40, 0.10),
            PreprocessInfo(True, "face", None, (200, 200)),
        )
        self.assertEqual(ev.label, LABEL_MIXED_SIGNALS)
        self.assertNotIn(ev.label, STRONG_LABELS)

    def test_fairness_extreme_does_not_force_mixed_if_main_agree(self) -> None:
        """0.72 / 0.68 main agree; fairness=0.05 should NOT cause Mixed signals."""
        ev = build_evaluation(
            ModelOutputs(0.72, 0.68, 0.05),
            PreprocessInfo(True, "face", None, (200, 200)),
        )
        self.assertNotEqual(ev.label, LABEL_MIXED_SIGNALS)

    def test_seven_four_one_main_spread_only(self) -> None:
        """0.7 vs 0.4 spread=0.30, straddle 0.5 → Mixed (fairness 0.1 ignored for spread)."""
        d = compute_disagreement(0.7, 0.4, 0.1)
        self.assertAlmostEqual(d.spread, 0.30, places=3)
        self.assertTrue(d.high_disagreement)
        ev = build_evaluation(
            ModelOutputs(0.7, 0.4, 0.1),
            PreprocessInfo(True, "face", None, (200, 200)),
        )
        self.assertEqual(ev.label, LABEL_MIXED_SIGNALS)
        self.assertNotEqual(ev.label, LABEL_LIKELY_AUTHENTIC)

    def test_moderate_spread_726_vs_484(self) -> None:
        d = compute_disagreement(0.726, 0.484, 0.113)
        self.assertAlmostEqual(d.spread, 0.242, places=3)
        self.assertEqual(d.tier, "moderate")

    def test_strong_fake_when_main_aligned(self) -> None:
        ev = build_evaluation(
            ModelOutputs(0.92, 0.88, 0.50),
            PreprocessInfo(True, "face", None, (224, 224)),
        )
        self.assertEqual(ev.label, LABEL_LIKELY_DEEPFAKE)


if __name__ == "__main__":
    unittest.main(verbosity=2)
