"""
Generate ensemble debug charts for one image (developer use).

Usage:
  cd unmask-backend
  set UNMASK_DEBUG=1
  venv\\Scripts\\python.exe scripts\\debug_visualize.py path\\to\\image.jpg

Requires matplotlib (pip install matplotlib).
"""

from __future__ import annotations

import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from PIL import Image

from model import predict_deepfake


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/debug_visualize.py <image_path> [out_dir]")
        sys.exit(1)

    path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(_ROOT, "debug_output")
    os.makedirs(out_dir, exist_ok=True)

    label, prob_fake, conf, explanation, _, report = predict_deepfake(Image.open(path).convert("RGB"))
    dbg = report.get("debug_scores", {})

    print("Label:", label)
    print("Prob fake:", prob_fake)
    print("Explanation:", explanation)
    print("\nDeveloper summary:")
    for line in report.get("developer_summary_lines", []):
        print(" ", line)

    json_path = os.path.join(out_dir, "report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("\nWrote", json_path)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Install matplotlib for charts: pip install matplotlib")
        return

    names = ["EffB4", "Xception", "Fairness", "Weighted", "Final"]
    w = dbg.get("weighted_before_fusion", dbg.get("raw_ensemble", 0))
    vals = [
        dbg.get("effb4", 0),
        dbg.get("xception", 0),
        dbg.get("fairness", 0),
        w,
        dbg.get("prob_fake", prob_fake),
    ]
    colors = ["#3b82f6", "#8b5cf6", "#10b981", "#f59e0b", "#ef4444"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(names, vals, color=colors)
    axes[0].axhline(0.5, color="gray", linestyle="--", alpha=0.7)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("P(fake)")
    axes[0].set_title("Ensemble pipeline")

    weights = dbg.get("weights", {})
    wnames = ["effb4", "xception", "fairness"]
    wvals = [weights.get(k, 0) for k in wnames]
    axes[1].pie(wvals, labels=wnames, autopct="%1.0f%%", colors=colors[:3])
    axes[1].set_title("Ensemble weights")

    plt.tight_layout()
    chart_path = os.path.join(out_dir, "ensemble_chart.png")
    plt.savefig(chart_path, dpi=120)
    plt.close()
    print("Wrote", chart_path)


if __name__ == "__main__":
    main()
