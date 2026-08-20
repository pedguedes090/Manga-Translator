"""Stress the vision pipeline on real comic pages with proposal bboxes."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import sys
from time import perf_counter

import cv2
import numpy as np

# Support the documented ``python tools/stress_test_comic_pages.py`` entrypoint.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from vision.config import VisionConfig
from vision.pipeline import VisionPipeline


def run_manifest(
    manifest: str | Path,
    config_path: str | Path,
    backend: str,
    erase_pages: int = 0,
) -> dict[str, object]:
    """Run every page independently and retain errors as report rows."""
    if erase_pages < 0:
        raise ValueError("erase_pages must not be negative")
    if backend not in {"heuristic", "hybrid"}:
        raise ValueError("stress backend must be heuristic or hybrid")

    manifest_path = Path(manifest)
    manifest_bytes = manifest_path.read_bytes()
    rows = [
        json.loads(line)
        for line in manifest_bytes.decode("utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("stress manifest is empty")

    config = replace(VisionConfig.load(config_path), mask_backend=backend)
    pipeline = VisionPipeline(config=config)
    report_rows: list[dict[str, object]] = []
    method_counts: Counter[str] = Counter()
    decision_counts: Counter[str] = Counter()
    decision_method_counts: Counter[str] = Counter()
    style_counts: dict[str, Counter[str]] = defaultdict(Counter)
    runtimes: list[float] = []
    successful_pages = 0
    failed_pages = 0
    block_count = 0

    for page_index, row in enumerate(rows):
        page_id = str(row.get("id", f"page-{page_index:06d}"))
        style = str(row.get("style", "unknown"))
        try:
            image = _decode_image(Path(str(row["image"])))
            if image is None:
                raise ValueError("could not decode image")
            blocks = _normalize_blocks(row.get("blocks", []))
            started = perf_counter()
            prepared = pipeline.prepare_page(image, blocks)
            per_page_methods = Counter(block.erase_method for block in prepared)
            per_page_decisions = Counter(block.decision.reason for block in prepared)
            method_counts.update(per_page_methods)
            decision_counts.update(per_page_decisions)
            decision_method_counts.update(
                f"{block.decision.reason}|{block.erase_method}" for block in prepared
            )
            block_count += len(prepared)

            outside_mask_delta: int | None = None
            changed_pixels: int | None = None
            warning_count = 0
            if page_index < erase_pages:
                before = image.copy()
                after = image.copy()
                union_mask = _union_mask(prepared, image.shape[:2])
                changed_pixels = 0
                for prepared_block in prepared:
                    erase_result = pipeline.erase_block(after, prepared_block)
                    changed_pixels += erase_result.changed_pixels
                    warning_count += int(erase_result.warning is not None)
                outside = union_mask == 0
                outside_mask_delta = int(
                    np.count_nonzero(before[outside] != after[outside])
                )

            elapsed_ms = (perf_counter() - started) * 1000.0
            runtimes.append(elapsed_ms)
            successful_pages += 1
            style_counts[style]["ok"] += 1
            coverages = [block.mask_result.coverage for block in prepared]
            edge_touches = [block.mask_result.edge_touch_ratio for block in prepared]
            report_rows.append(
                {
                    "block_count": len(prepared),
                    "changed_pixels": changed_pixels,
                    "decision_counts": dict(sorted(per_page_decisions.items())),
                    "elapsed_ms": elapsed_ms,
                    "id": page_id,
                    "mean_coverage": float(np.mean(coverages)) if coverages else 0.0,
                    "mean_edge_touch_ratio": (
                        float(np.mean(edge_touches)) if edge_touches else 0.0
                    ),
                    "method_counts": dict(sorted(per_page_methods.items())),
                    "outside_mask_delta": outside_mask_delta,
                    "page_class": str(row.get("page_class", "unknown")),
                    "status": "ok",
                    "style": style,
                    "warning_count": warning_count,
                }
            )
        except Exception as exc:  # Per-page isolation is a benchmark requirement.
            failed_pages += 1
            style_counts[style]["error"] += 1
            report_rows.append(
                {
                    "error": _safe_error(exc),
                    "id": page_id,
                    "status": "error",
                    "style": style,
                }
            )

    return {
        "annotation_semantics": sorted(
            {str(row.get("annotation_semantics", "unknown")) for row in rows}
        ),
        "backend": backend,
        "block_count": block_count,
        "config_hash": config.config_hash(),
        "dataset_hash": sha256(manifest_bytes).hexdigest(),
        "erase_pages": min(erase_pages, len(rows)),
        "failed_pages": failed_pages,
        "method_counts": dict(sorted(method_counts.items())),
        "decision_counts": dict(sorted(decision_counts.items())),
        "decision_method_counts": dict(sorted(decision_method_counts.items())),
        "page_count": len(rows),
        "rows": report_rows,
        "runtime": {
            "page_p50_ms": float(np.percentile(runtimes, 50)) if runtimes else None,
            "page_p95_ms": float(np.percentile(runtimes, 95)) if runtimes else None,
            "page_total_ms": float(sum(runtimes)),
        },
        "runtime_provider": "opencv-cpu",
        "styles": {
            style: dict(sorted(counts.items()))
            for style, counts in sorted(style_counts.items())
        },
        "successful_pages": successful_pages,
    }


def _decode_image(path: Path) -> np.ndarray | None:
    try:
        encoded = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR) if encoded.size else None


def _normalize_blocks(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        raise ValueError("blocks must be a list")
    blocks: list[dict[str, object]] = []
    for block in value:
        if not isinstance(block, dict) or "bbox" not in block:
            raise ValueError("each block must contain a bbox")
        blocks.append({"bbox": block["bbox"], "text": str(block.get("text", ""))})
    return blocks


def _union_mask(prepared: list[object], image_shape: tuple[int, int]) -> np.ndarray:
    union = np.zeros(image_shape, np.uint8)
    for block in prepared:
        mask_result = block.mask_result
        x1, y1, x2, y2 = mask_result.roi_bbox
        union[y1:y2, x1:x2] = cv2.bitwise_or(
            union[y1:y2, x1:x2], mask_result.mask
        )
    return union


def _safe_error(exc: Exception) -> str:
    if isinstance(exc, ValueError) and str(exc) == "could not decode image":
        return "could not decode image"
    return type(exc).__name__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--backend", required=True, choices=("heuristic", "hybrid"))
    parser.add_argument("--erase-pages", type=int, default=0)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = run_manifest(
        args.manifest,
        args.config,
        backend=args.backend,
        erase_pages=args.erase_pages,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
