"""Build a deterministic, local-only manifest of real comic pages."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import io
import json
from pathlib import Path
from typing import Iterable

from PIL import Image, UnidentifiedImageError


SUPPORTED_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


@dataclass(frozen=True)
class ComicSource:
    """A labelled local directory whose files remain outside Git."""

    name: str
    style: str
    root: Path
    license: str


def build_manifest(
    sources: Iterable[ComicSource],
    output: str | Path,
    min_side: int = 64,
) -> dict[str, object]:
    """Validate, de-duplicate, and write comic page metadata as JSONL."""
    if min_side < 1:
        raise ValueError("min_side must be positive")
    output_path = Path(output)
    rows: list[dict[str, object]] = []
    seen_hashes: set[str] = set()
    candidate_count = 0
    duplicate_count = 0
    invalid_count = 0

    for source in sources:
        root = Path(source.root).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"comic source root is not a directory: {root}")
        if not source.name.strip() or not source.style.strip() or not source.license.strip():
            raise ValueError("comic source name, style, and license must be non-empty")

        candidates = sorted(
            (
                path
                for path in root.rglob("*")
                if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
            ),
            key=lambda path: str(path).casefold(),
        )
        candidate_count += len(candidates)

        for path in candidates:
            try:
                payload = path.read_bytes()
            except OSError:
                invalid_count += 1
                continue
            digest = sha256(payload).hexdigest()
            if digest in seen_hashes:
                duplicate_count += 1
                continue
            try:
                with Image.open(io.BytesIO(payload)) as image:
                    image.load()
                    width, height = image.size
            except (OSError, ValueError, UnidentifiedImageError):
                invalid_count += 1
                continue
            if width < min_side or height < min_side:
                invalid_count += 1
                continue

            seen_hashes.add(digest)
            relative_parent = path.relative_to(root).parent
            group = relative_parent.as_posix() if relative_parent.parts else "."
            rows.append(
                {
                    "group": group,
                    "height": height,
                    "id": f"{source.name}-{digest[:16]}",
                    "image": str(path.resolve()),
                    "license": source.license,
                    "sha256": digest,
                    "source": source.name,
                    "split": "stress",
                    "style": source.style,
                    "width": width,
                }
            )

    rows.sort(key=lambda row: (str(row["source"]), str(row["image"]).casefold()))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )
    return {
        "candidate_count": candidate_count,
        "duplicate_count": duplicate_count,
        "invalid_count": invalid_count,
        "page_count": len(rows),
        "sources": dict(sorted(Counter(row["source"] for row in rows).items())),
        "styles": dict(sorted(Counter(row["style"] for row in rows).items())),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        nargs=4,
        metavar=("NAME", "STYLE", "LICENSE", "ROOT"),
        required=True,
        help="repeat for each local source directory",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-side", type=int, default=64)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    sources = [
        ComicSource(name=name, style=style, license=license_name, root=Path(root))
        for name, style, license_name, root in args.source
    ]
    summary = build_manifest(sources, args.output, min_side=args.min_side)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
