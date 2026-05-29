"""Decision gating — policy v3.0 (main spread only, no unanimity)."""

from __future__ import annotations

import os
import sys
import unittest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from evaluation import (
    LABEL_INSUFFICIENT_FACE,
    LABEL_LIKELY_AUTHENTIC,
    LABEL_LIKELY_DEEPFAKE,
    LABEL_MIXED_SIGNALS,
    STRONG_LABELS,
    ModelOutputs,
    PreprocessInfo,
    build_evaluation,
)


class TestDecisionGating(unittest.TestCase):
    def test_no_face_insufficient_evidence(self) -> None:
        ev = build_evaluation(
            ModelOutputs(0.726, 0.484, 0.113),
            PreprocessInfo(False, "no face", None, None),
        )
        self.assertEqual(ev.label, LABEL_INSUFFICIENT_FACE)

    def test_high_main_spread_mixed_not_authentic(self) -> None:
        ev = build_evaluation(
            ModelOutputs(0.7, 0.4, 0.1),
            PreprocessInfo(True, "face", None, (224, 224)),
        )
        self.assertEqual(ev.label, LABEL_MIXED_SIGNALS)
        self.assertNotIn(ev.label, STRONG_LABELS)

    def test_strong_authentic_low_scores(self) -> None:
        ev = build_evaluation(
            ModelOutputs(0.08, 0.12, 0.05),
            PreprocessInfo(True, "face", None, (224, 224)),
        )
        self.assertEqual(ev.label, LABEL_LIKELY_AUTHENTIC)

    def test_explanation_no_raw_debug(self) -> None:
        ev = build_evaluation(
            ModelOutputs(0.7, 0.4, 0.1),
            PreprocessInfo(True, "face", None, (224, 224)),
        )
        self.assertNotIn("effb4=", ev.user_explanation.lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
