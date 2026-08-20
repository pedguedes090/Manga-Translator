"""Generate deterministic synthetic text-mask evaluation samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


_CATEGORY_CYCLE = (
    ["white_bubble"] * 4
    + ["dark_bubble"] * 2
    + ["colored_bubble"] * 2
    + ["outlined_text"] * 3
    + ["screentone"] * 3
    + ["complex_artwork"] * 3
    + ["sfx_outside_bubble"] * 2
    + ["clipped_bbox"]
)
_LANGUAGES = ("ja", "ko", "zh", "en", "vi")


def generate_dataset(output: str | Path, samples: int, seed: int) -> Path:
    """Generate samples and return the JSONL manifest path."""
    if samples < 1:
        raise ValueError("samples must be positive")

    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []

    for index in range(samples):
        category = _CATEGORY_CYCLE[index % len(_CATEGORY_CYCLE)]
        image, clean, text_mask, bubble_mask, bbox = _render_sample(category, rng)
        names = {
            "image": f"image_{index:04d}.png",
            "clean_target": f"clean_target_{index:04d}.png",
            "text_mask": f"text_mask_{index:04d}.png",
            "bubble_mask": f"bubble_mask_{index:04d}.png",
        }
        _write_png(output_path / names["image"], image)
        _write_png(output_path / names["clean_target"], clean)
        _write_png(output_path / names["text_mask"], text_mask)
        _write_png(output_path / names["bubble_mask"], bubble_mask)
        rows.append(
            {
                "id": f"synthetic-{index:04d}",
                **names,
                "bbox": list(bbox),
                "text": "TEXT",
                "category": category,
                "language": _LANGUAGES[index % len(_LANGUAGES)],
                "source": "project-synthetic",
                "license": "project",
                "split": "test",
            }
        )

    manifest = output_path / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return manifest


def _render_sample(
    category: str, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    height, width = 192, 256
    clean = np.full((height, width, 3), 235, np.uint8)
    bubble_mask = np.zeros((height, width), np.uint8)

    if category == "white_bubble":
        cv2.ellipse(clean, (128, 96), (102, 62), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(clean, (128, 96), (102, 62), 0, 0, 360, (20, 20, 20), 3)
        cv2.ellipse(bubble_mask, (128, 96), (99, 59), 0, 0, 360, 255, -1)
        text_color = (0, 0, 0)
    elif category == "dark_bubble":
        cv2.ellipse(clean, (128, 96), (102, 62), 0, 0, 360, (35, 35, 35), -1)
        cv2.ellipse(clean, (128, 96), (102, 62), 0, 0, 360, (235, 235, 235), 3)
        cv2.ellipse(bubble_mask, (128, 96), (99, 59), 0, 0, 360, 255, -1)
        text_color = (255, 255, 255)
    elif category in {"colored_bubble", "outlined_text"}:
        color = tuple(int(value) for value in rng.integers(145, 210, size=3))
        cv2.rectangle(clean, (24, 34), (232, 158), color, -1)
        cv2.rectangle(bubble_mask, (27, 37), (229, 155), 255, -1)
        text_color = (0, 0, 0)
    elif category == "screentone":
        for y in range(4, height, 8):
            for x in range(4, width, 8):
                cv2.circle(clean, (x, y), 1, (120, 120, 120), -1)
        text_color = (0, 0, 0)
    else:
        clean[:] = rng.integers(90, 220, size=(height, width, 1), dtype=np.uint8)
        for offset in range(-height, width, 18):
            cv2.line(clean, (max(offset, 0), max(-offset, 0)),
                     (min(width - 1, offset + height), min(height - 1, height + offset)),
                     (35, 55, 75), 2)
        text_color = (255, 255, 255) if category == "sfx_outside_bubble" else (0, 0, 0)

    text_mask = np.zeros((height, width), np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    origin = (65, 108)
    if category == "outlined_text":
        cv2.putText(text_mask, "TEXT", origin, font, 1.25, 255, 7, cv2.LINE_AA)
        image = clean.copy()
        cv2.putText(image, "TEXT", origin, font, 1.25, (255, 255, 255), 7, cv2.LINE_AA)
        cv2.putText(image, "TEXT", origin, font, 1.25, text_color, 3, cv2.LINE_AA)
    else:
        cv2.putText(text_mask, "TEXT", origin, font, 1.25, 255, 3, cv2.LINE_AA)
        image = clean.copy()
        cv2.putText(image, "TEXT", origin, font, 1.25, text_color, 3, cv2.LINE_AA)

    ys, xs = np.nonzero(text_mask)
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)
    if category == "clipped_bbox":
        x1 += 10
        x2 -= 8
    else:
        x1, y1 = max(0, x1 - 6), max(0, y1 - 6)
        x2, y2 = min(width, x2 + 6), min(height, y2 + 6)
    return image, clean, text_mask, bubble_mask, (x1, y1, x2, y2)


def _write_png(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise OSError(f"could not write {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--samples", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    generate_dataset(args.output, args.samples, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
