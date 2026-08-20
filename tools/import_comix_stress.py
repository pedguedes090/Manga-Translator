"""Normalize COMIX page annotations for local-only stress testing."""

from __future__ import annotations

import argparse
from hashlib import sha256
import io
import json
from pathlib import Path

from PIL import Image, UnidentifiedImageError


def import_comix(
    source: str | Path,
    output: str | Path,
    score_threshold: float = 0.25,
) -> dict[str, int]:
    """Write COMIX detector proposals without treating them as ground truth."""
    if not 0.0 <= score_threshold <= 1.0:
        raise ValueError("score_threshold must be between zero and one")
    source_path = Path(source).resolve()
    if not source_path.is_dir():
        raise ValueError(f"COMIX source is not a directory: {source_path}")

    rows: list[dict[str, object]] = []
    block_count = 0
    invalid_page_count = 0
    pages_without_text = 0
    for annotation_path in sorted(source_path.glob("*.json"), key=lambda path: path.name):
        try:
            annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
            image_name = str(annotation["image"]["file"])
            image_path = (source_path / image_name).resolve()
            image_path.relative_to(source_path)
            payload = image_path.read_bytes()
            with Image.open(io.BytesIO(payload)) as image:
                image.load()
                width, height = image.size
            blocks = _text_blocks(annotation, width, height, score_threshold)
        except (
            KeyError,
            OSError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
            UnidentifiedImageError,
        ):
            invalid_page_count += 1
            continue

        if not blocks:
            pages_without_text += 1
        block_count += len(blocks)
        rows.append(
            {
                "annotation_semantics": "detector-proposal",
                "blocks": blocks,
                "book_id": str(annotation.get("book_id", "")),
                "height": height,
                "id": str(annotation.get("page_id", annotation_path.stem)),
                "image": str(image_path),
                "license": "CC0-1.0",
                "page_class": str(annotation.get("page_class", "unknown")),
                "page_number": int(annotation.get("page_number", -1)),
                "sha256": sha256(payload).hexdigest(),
                "source": "comix-v0-tiny-pages",
                "split": "stress",
                "style": "comic",
                "width": width,
            }
        )

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )
    return {
        "block_count": block_count,
        "invalid_page_count": invalid_page_count,
        "page_count": len(rows),
        "pages_without_text": pages_without_text,
    }


def _text_blocks(
    annotation: dict[str, object],
    width: int,
    height: int,
    score_threshold: float,
) -> list[dict[str, object]]:
    detections = annotation.get("detections", {})
    if not isinstance(detections, dict):
        return []
    blocks: list[dict[str, object]] = []
    for detector in ("magi", "fasterrcnn"):
        detector_rows = detections.get(detector, {})
        if not isinstance(detector_rows, dict):
            continue
        proposals = detector_rows.get("text", [])
        if not isinstance(proposals, list):
            continue
        for proposal in proposals:
            if not isinstance(proposal, dict):
                continue
            score = float(proposal.get("score", 1.0))
            if score < score_threshold:
                continue
            bbox = _clamp_bbox(proposal.get("bbox"), width, height)
            if bbox is not None:
                blocks.append({"bbox": bbox, "detector": detector, "score": score})
        if blocks:
            break
    return blocks


def _clamp_bbox(value: object, width: int, height: int) -> list[int] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    x1, y1, x2, y2 = (int(round(float(item))) for item in value[:4])
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    clamped = [
        max(0, min(width, x1)),
        max(0, min(height, y1)),
        max(0, min(width, x2)),
        max(0, min(height, y2)),
    ]
    return clamped if clamped[2] > clamped[0] and clamped[3] > clamped[1] else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--score-threshold", type=float, default=0.25)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = import_comix(args.source, args.output, args.score_threshold)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
