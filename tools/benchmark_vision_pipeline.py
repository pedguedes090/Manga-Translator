"""Run paired legacy/prepared image erasure benchmarks."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Mapping

import cv2
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from add_text import appearance_for_prepared, erase_text_region, render_all_blocks
from vision.config import VisionConfig
from vision.metrics import (
    compute_bubble_border_damage,
    compute_inpainting_metrics,
    compute_mask_metrics,
    compute_outside_mask_delta,
)
from vision.pipeline import VisionPipeline


_ALLOWED_MODES = {"legacy", "prepared"}
_ALLOWED_BACKENDS = {"heuristic", "hybrid"}


def benchmark_manifest(
    manifest_path: str | Path,
    config_path: str | Path,
    mode: str,
    backend: str,
    *,
    indices: list[int] | None = None,
    warmup: int = 0,
    pipeline: object | None = None,
) -> dict[str, object]:
    """Benchmark identical manifest rows through one erasure mode."""
    if mode not in _ALLOWED_MODES:
        raise ValueError("mode must be legacy or prepared")
    if backend not in _ALLOWED_BACKENDS:
        raise ValueError("backend must be heuristic or hybrid")
    if warmup < 0:
        raise ValueError("warmup must not be negative")

    manifest = Path(manifest_path)
    manifest_bytes = manifest.read_bytes()
    rows = [
        json.loads(line)
        for line in manifest_bytes.decode("utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("benchmark manifest is empty")

    base_config = VisionConfig.load(config_path)
    config = replace(base_config, mask_backend=backend)
    active_pipeline = pipeline
    if mode == "prepared" and active_pipeline is None:
        active_pipeline = VisionPipeline(config=config)

    selected = _select_indices(len(rows), indices)
    if warmup:
        _warmup(active_pipeline, mode, rows, manifest.parent, selected[0], warmup)

    report_rows: list[dict[str, object]] = []
    timing_values: dict[str, list[float]] = {
        "decode_ms": [],
        "prepare_ms": [],
        "erase_ms": [],
        "render_ms": [],
        "total_ms": [],
    }
    method_counts: Counter[str] = Counter()
    warning_count = 0
    failed_pages = 0

    for row_index in selected:
        row = rows[row_index]
        page_id = str(row.get("id", f"page-{row_index:06d}"))
        started_total = perf_counter()
        decode_ms = prepare_ms = erase_ms = render_ms = 0.0
        try:
            started = perf_counter()
            image = _decode_image(manifest.parent / str(row["image"]))
            decode_ms = _elapsed_ms(started)
            blocks = _normalize_blocks(row)
            before = image.copy()
            appearances: list[dict[str, object]] = []
            prepared: list[object] = []
            erase_results: list[object] = []

            if mode == "prepared":
                if active_pipeline is None:
                    raise RuntimeError("prepared benchmark requires a pipeline")
                started = perf_counter()
                prepared = list(active_pipeline.prepare_page(image, blocks))
                prepare_ms = _elapsed_ms(started)
                started = perf_counter()
                erased, erase_results = active_pipeline.erase_page(image, prepared)
                erase_ms = _elapsed_ms(started)
                image = erased
                if len(erase_results) != len(prepared):
                    raise RuntimeError(
                        "prepared benchmark received mismatched erase results"
                    )
                appearances = [_prepared_appearance(item) for item in prepared]
                methods = [str(result.method) for result in erase_results]
            else:
                started = perf_counter()
                methods = []
                for block in blocks:
                    image, text_color, appearance = erase_text_region(
                        image,
                        block["bbox"],
                        source_lang=str(row.get("language", "ja")),
                    )
                    appearance = dict(appearance)
                    appearance["should_skip"] = False
                    appearance["text_color"] = text_color
                    appearances.append(appearance)
                    methods.append(str(appearance.get("erase_method", "legacy")))
                erase_ms = _elapsed_ms(started)

            method_counts.update(methods)
            row_warning_count = sum(
                bool(getattr(result, "warning", None)) for result in erase_results
            ) + sum(bool(item.get("erase_warning")) for item in appearances)
            warning_count += row_warning_count

            render_blocks = _render_blocks(blocks, appearances)
            started = perf_counter()
            render_all_blocks(image.copy(), render_blocks, None)
            render_ms = _elapsed_ms(started)
            timing = {
                "decode_ms": decode_ms,
                "prepare_ms": prepare_ms,
                "erase_ms": erase_ms,
                "render_ms": render_ms,
                "total_ms": _elapsed_ms(started_total),
            }
            for key, value in timing.items():
                timing_values[key].append(value)
            quality = _quality_metrics(
                row=row,
                root=manifest.parent,
                before=before,
                after=image,
                prepared=prepared,
                mode=mode,
            )
            report_rows.append(
                {
                    "id": page_id,
                    "status": "ok",
                    "block_count": len(blocks),
                    "method_counts": dict(sorted(Counter(methods).items())),
                    "warning_count": row_warning_count,
                    "rendered_block_count": len(render_blocks),
                    **timing,
                    **quality,
                }
            )
        except Exception as exc:
            failed_pages += 1
            report_rows.append(
                {
                    "id": page_id,
                    "status": "error",
                    "error": _safe_error(exc),
                    "decode_ms": decode_ms,
                    "prepare_ms": prepare_ms,
                    "erase_ms": erase_ms,
                    "render_ms": render_ms,
                    "total_ms": _elapsed_ms(started_total),
                }
            )

    return {
        "schema_version": 1,
        "mode": mode,
        "backend": backend,
        "annotation_semantics": _annotation_semantics(rows),
        "dataset_hash": sha256(manifest_bytes).hexdigest(),
        "config_hash": config.config_hash(),
        "selected_count": len(selected),
        "successful_pages": len(selected) - failed_pages,
        "failed_pages": failed_pages,
        "warning_count": warning_count,
        "method_counts": dict(sorted(method_counts.items())),
        "runtime": _runtime_capabilities(),
        "summary": {
            key: _timing_summary(values)
            for key, values in timing_values.items()
        },
        "rows": report_rows,
    }


def _select_indices(row_count: int, indices: list[int] | None) -> list[int]:
    selected = list(range(row_count)) if indices is None else list(indices)
    if not selected:
        raise ValueError("indices must not be empty")
    if len(set(selected)) != len(selected):
        raise ValueError("indices must be unique")
    if any(index < 0 or index >= row_count for index in selected):
        raise IndexError("benchmark index is outside the manifest")
    return selected


def _warmup(
    pipeline: object | None,
    mode: str,
    rows: list[dict[str, object]],
    root: Path,
    row_index: int,
    count: int,
) -> None:
    row = rows[row_index]
    image = _decode_image(root / str(row["image"]))
    blocks = _normalize_blocks(row)
    for _ in range(count):
        if mode == "prepared":
            if pipeline is None:
                raise RuntimeError("prepared benchmark requires a pipeline")
            prepared = list(pipeline.prepare_page(image, blocks))
            pipeline.erase_page(image.copy(), prepared)
        else:
            probe = image.copy()
            for block in blocks:
                erase_text_region(probe, block["bbox"], source_lang=str(row.get("language", "ja")))


def _decode_image(path: Path) -> np.ndarray:
    try:
        encoded = np.fromfile(path, dtype=np.uint8)
    except OSError as exc:
        raise ValueError("could not decode image") from exc
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR) if encoded.size else None
    if image is None:
        raise ValueError("could not decode image")
    return image


def _read_optional(path: Path, mode: int) -> np.ndarray | None:
    if not path.exists():
        return None
    image = cv2.imread(str(path), mode)
    if image is None:
        raise ValueError(f"could not read annotation {path.name}")
    return image


def _normalize_blocks(row: Mapping[str, object]) -> list[dict[str, object]]:
    raw_blocks = row.get("blocks")
    if raw_blocks is None:
        raw_blocks = [{"bbox": row.get("bbox"), "text": row.get("text", "")}]
    if not isinstance(raw_blocks, list):
        raise ValueError("blocks must be a list")
    blocks: list[dict[str, object]] = []
    for block in raw_blocks:
        if not isinstance(block, dict) or "bbox" not in block:
            raise ValueError("each block must contain a bbox")
        bbox = block["bbox"]
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            raise ValueError("each block bbox must contain four coordinates")
        blocks.append({"bbox": list(bbox[:4]), "text": str(block.get("text", ""))})
    return blocks


def _prepared_appearance(prepared: object) -> dict[str, object]:
    return dict(appearance_for_prepared(prepared))


def _render_blocks(
    blocks: list[dict[str, object]],
    appearances: list[dict[str, object]],
) -> list[dict[str, object]]:
    rendered: list[dict[str, object]] = []
    for block, appearance in zip(blocks, appearances):
        text = str(block.get("text", "")).strip()
        if text and not appearance.get("should_skip", False):
            rendered.append(
                {
                    "text": text,
                    "bbox": block["bbox"],
                    "text_color": appearance.get("text_color", (0, 0, 0)),
                    "appearance": appearance,
                }
            )
    return rendered


def _quality_metrics(
    *,
    row: Mapping[str, object],
    root: Path,
    before: np.ndarray,
    after: np.ndarray,
    prepared: list[object],
    mode: str,
) -> dict[str, object]:
    clean_target = _read_optional(root / str(row["clean_target"]), cv2.IMREAD_COLOR) if row.get("clean_target") else None
    target_mask = _read_optional(root / str(row["text_mask"]), cv2.IMREAD_GRAYSCALE) if row.get("text_mask") else None
    if clean_target is None or target_mask is None:
        return {"quality_status": "not_available"}
    if clean_target.shape != before.shape or target_mask.shape != before.shape[:2]:
        raise ValueError("quality annotations do not match image dimensions")

    predicted_mask = _union_prepared(prepared, before.shape[:2]) if mode == "prepared" else _changed_mask(before, after)
    mask_metrics = compute_mask_metrics(predicted_mask, target_mask)
    inpainting = compute_inpainting_metrics(before, after, clean_target, target_mask)
    inpainting["outside_reference_mask_delta"] = float(
        compute_outside_mask_delta(before, after, target_mask)
    )
    inpainting["outside_predicted_mask_delta"] = (
        float(compute_outside_mask_delta(before, after, predicted_mask))
        if mode == "prepared"
        else None
    )
    bubble_path = row.get("bubble_mask")
    if bubble_path:
        bubble_mask = _read_optional(root / str(bubble_path), cv2.IMREAD_GRAYSCALE)
        if bubble_mask is not None:
            bubble_boundary = _mask_boundary(bubble_mask)
            inpainting["bubble_border_damage"] = compute_bubble_border_damage(
                before, after, bubble_boundary
            )
    return {
        "quality_status": "measured",
        "mask_semantics": "prepared-prediction" if mode == "prepared" else "changed-pixels-proxy",
        "mask_metrics": mask_metrics,
        "inpainting_metrics": inpainting,
    }


def _union_prepared(prepared: list[object], shape: tuple[int, int]) -> np.ndarray:
    union = np.zeros(shape, dtype=np.uint8)
    for block in prepared:
        x1, y1, x2, y2 = block.mask_result.roi_bbox
        union[y1:y2, x1:x2] = cv2.bitwise_or(
            union[y1:y2, x1:x2], block.mask_result.mask
        )
    return union


def _changed_mask(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    return np.any(before != after, axis=2).astype(np.uint8) * 255


def _mask_boundary(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    eroded = cv2.erode(binary, np.ones((3, 3), np.uint8), iterations=1)
    return ((binary > 0) & (eroded == 0)).astype(np.uint8) * 255


def _annotation_semantics(rows: list[Mapping[str, object]]) -> list[str]:
    values = {
        str(row.get("annotation_semantics", "synthetic-clean-target" if row.get("clean_target") else "detector-proposal"))
        for row in rows
    }
    return sorted(values)


def _elapsed_ms(started: float) -> float:
    return (perf_counter() - started) * 1000.0


def _timing_summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"count": 0.0, "mean_ms": None, "p50_ms": None, "p95_ms": None, "p99_ms": None}
    return {
        "count": float(len(values)),
        "mean_ms": float(np.mean(values)),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
    }


def _runtime_capabilities() -> dict[str, object]:
    runtime: dict[str, object] = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "torch_available": False,
        "cuda_available": False,
        "cuda_device_count": 0,
    }
    try:
        import torch
    except ImportError:
        return runtime
    runtime["torch_available"] = True
    runtime["cuda_available"] = bool(torch.cuda.is_available())
    runtime["cuda_device_count"] = int(torch.cuda.device_count())
    if runtime["cuda_available"]:
        runtime["cuda_device"] = str(torch.cuda.get_device_name(0))
    return runtime


def _safe_error(exc: Exception) -> str:
    if isinstance(exc, ValueError) and str(exc) == "could not decode image":
        return "could not decode image"
    return type(exc).__name__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=sorted(_ALLOWED_MODES))
    parser.add_argument("--backend", required=True, choices=sorted(_ALLOWED_BACKENDS))
    parser.add_argument("--indices", type=int, nargs="*")
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = benchmark_manifest(
        args.manifest,
        args.config,
        mode=args.mode,
        backend=args.backend,
        indices=args.indices,
        warmup=args.warmup,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
