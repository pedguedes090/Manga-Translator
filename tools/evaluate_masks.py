"""Evaluate binary text-mask predictions against a JSONL manifest."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Callable

import cv2
import numpy as np

# Support the documented ``python tools/evaluate_masks.py`` entrypoint.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from vision.metrics import compute_mask_metrics


MaskBackend = Callable[[np.ndarray, tuple[int, int, int, int], str], np.ndarray]


def evaluate_manifest(
    manifest_path: str | Path,
    backend: MaskBackend,
    config_hash: str,
) -> dict[str, object]:
    manifest = Path(manifest_path)
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError("evaluation manifest is empty")

    metric_rows: list[dict[str, object]] = []
    by_category: dict[str, list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        image = _read_image(manifest.parent / row["image"], cv2.IMREAD_COLOR)
        target = _read_image(manifest.parent / row["text_mask"], cv2.IMREAD_GRAYSCALE)
        bbox = tuple(int(value) for value in row["bbox"])
        prediction = backend(image, bbox, str(row.get("text", "")))
        metrics = compute_mask_metrics(prediction, target)
        category = str(row["category"])
        by_category[category].append(metrics)
        metric_rows.append({"id": row["id"], "category": category, **metrics})

    metric_names = tuple(key for key in metric_rows[0] if key not in {"id", "category"})
    summary = _mean_metrics(metric_rows, metric_names)
    category_summary = {
        category: _mean_metrics(values, metric_names)
        for category, values in sorted(by_category.items())
    }
    return {
        "config_hash": config_hash,
        "sample_count": len(metric_rows),
        "summary": summary,
        "categories": category_summary,
        "rows": metric_rows,
    }


def _read_image(path: Path, mode: int) -> np.ndarray:
    image = cv2.imread(str(path), mode)
    if image is None:
        raise ValueError(f"could not read image: {path}")
    return image


def _mean_metrics(rows: list[dict[str, object]], names: tuple[str, ...]) -> dict[str, float]:
    return {
        name: float(np.mean([float(row[name]) for row in rows]))
        for name in names
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--backend", required=True, choices=("heuristic", "hybrid", "neural"))
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def _load_backend(name: str, config_path: Path) -> tuple[MaskBackend, str]:
    from vision.config import VisionConfig

    config = VisionConfig.load(config_path)
    if name == "heuristic":
        from vision.maskers.heuristic import HeuristicTextMasker

        masker = HeuristicTextMasker(config.text_mask)
    elif name == "hybrid":
        from vision.maskers.hybrid import HybridTextMasker

        masker = HybridTextMasker(config.text_mask)
    else:
        from vision.maskers.neural import build_neural_text_masker

        masker = build_neural_text_masker(config)

    def predict(image: np.ndarray, bbox: tuple[int, int, int, int], text: str) -> np.ndarray:
        from vision.region_analysis import analyze_region

        result = masker.generate(image, bbox, text, analyze_region(image, bbox), None)
        full_mask = np.zeros(image.shape[:2], np.uint8)
        x1, y1, x2, y2 = result.roi_bbox
        full_mask[y1:y2, x1:x2] = result.mask
        return full_mask

    return predict, config.config_hash()


def main() -> int:
    args = build_parser().parse_args()
    backend, config_hash = _load_backend(args.backend, args.config)
    report = evaluate_manifest(args.manifest, backend, config_hash)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
