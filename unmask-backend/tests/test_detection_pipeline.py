"""
Regression tests for Unmask detection pipeline.
Run from unmask-backend: python -m pytest tests/test_detection_pipeline.py -v
Or: python tests/test_detection_pipeline.py
"""

from __future__ import annotations

import os
import sys
import unittest

# unmask-backend root on path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from PIL import Image

from evaluation import THRESHOLD_FAKE, THRESHOLD_REAL
from model import predict_deepfake


def _repo_root() -> str:
    return os.path.dirname(_ROOT)


def _fairness_data() -> tuple[str, str]:
    base = os.path.join(_repo_root(), "fairness_model", "data")
    fake = os.path.join(base, "fake", "openart-image__Vcu-xcx_1770431564188_raw.jpg")
    real = os.path.join(base, "real", "1_0_1_20161219160115237.jpg")
    return fake, real


class TestDetectionPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ["UNMASK_DETERMINISTIC"] = "1"
        fake_p, real_p = _fairness_data()
        if not os.path.isfile(fake_p):
            raise unittest.SkipTest(f"Missing fake sample: {fake_p}")
        if not os.path.isfile(real_p):
            raise unittest.SkipTest(f"Missing real sample: {real_p}")
        cls.fake_path = fake_p
        cls.real_path = real_p

    def test_fake_sample_high_prob_fake(self) -> None:
        label, prob_fake, overall_conf, expl, _, report = predict_deepfake(
            Image.open(self.fake_path).convert("RGB")
        )
        dbg = report["debug_scores"]
        self.assertGreaterEqual(prob_fake, 0.5, f"fake image prob_fake too low: {prob_fake} dbg={dbg}")
        self.assertNotIn("effb4=", expl.lower())
        self.assertIn(
            label,
            ("Likely deepfake", "Possibly AI-generated (novel pattern)", "Mixed signals", "Inconclusive"),
        )

    def test_real_sample_not_labeled_deepfake(self) -> None:
        """Real UTKFace Black sample should not be confidently classified as deepfake."""
        label, prob_fake, overall_conf, _, _, report = predict_deepfake(
            Image.open(self.real_path).convert("RGB")
        )
        dbg = report["debug_scores"]
        self.assertLess(
            prob_fake,
            THRESHOLD_FAKE,
            f"real image misclassified as fake: label={label} prob_fake={prob_fake} dbg={dbg}",
        )
        strong_fake = {"Likely deepfake"}
        self.assertNotIn(
            label,
            strong_fake,
            f"real sample must not get strong fake label: {label} prob={prob_fake}",
        )
        self.assertIn(
            label,
            (
                "Likely authentic",
                "Insufficient facial evidence",
                "Mixed signals",
                "Inconclusive",
                "Unable to analyze reliably",
                "Possibly AI-generated (novel pattern)",
            ),
        )

    def test_scores_in_valid_range(self) -> None:
        for path in (self.fake_path, self.real_path):
            label, prob_fake, overall_conf, _, _, report = predict_deepfake(
                Image.open(path).convert("RGB")
            )
            dbg = report["debug_scores"]
            self.assertGreaterEqual(prob_fake, 0.0)
            self.assertLessEqual(prob_fake, 1.0)
            self.assertGreaterEqual(overall_conf, 0.0)
            self.assertLessEqual(overall_conf, 1.0)
            self.assertIn("confidence_level", dbg)
            for key in ("effb4", "xception", "fairness", "ensemble"):
                self.assertIn(key, dbg)
                self.assertGreaterEqual(dbg[key], 0.0)
                self.assertLessEqual(dbg[key], 1.0)

    def test_deterministic_repeat(self) -> None:
        img = Image.open(self.real_path).convert("RGB")
        a = predict_deepfake(img)
        b = predict_deepfake(img)
        self.assertEqual(a[1], b[1], "prob_fake should be deterministic")
        self.assertEqual(a[0], b[0], "label should be deterministic")


if __name__ == "__main__":
    unittest.main(verbosity=2)
