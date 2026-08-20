import json
from pathlib import Path

import numpy as np
from PIL import Image

from tools.build_comic_stress_manifest import ComicSource, build_manifest


def _write_image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((80, 120, 3), value, np.uint8)
    Image.fromarray(image).save(path)


def test_manifest_filters_corrupt_and_duplicate_pages(tmp_path):
    root = tmp_path / "truyện Hàn"
    first = root / "chapter 1" / "001.png"
    duplicate = root / "chapter 2" / "copy.png"
    corrupt = root / "chapter 3" / "broken.png"
    tiny = root / "chapter 4" / "banner.png"
    _write_image(first, 210)
    duplicate.parent.mkdir(parents=True)
    duplicate.write_bytes(first.read_bytes())
    corrupt.parent.mkdir(parents=True)
    corrupt.write_bytes(b"not an image")
    tiny.parent.mkdir(parents=True)
    Image.fromarray(np.full((20, 120, 3), 128, np.uint8)).save(tiny)
    output = tmp_path / "manifest.jsonl"

    summary = build_manifest(
        [
            ComicSource(
                name="local-manhwa",
                style="manhwa",
                root=root,
                license="user-owned-local-evaluation",
            )
        ],
        output,
    )

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert summary == {
        "candidate_count": 4,
        "duplicate_count": 1,
        "invalid_count": 2,
        "page_count": 1,
        "sources": {"local-manhwa": 1},
        "styles": {"manhwa": 1},
    }
    assert rows == [
        {
            "group": "chapter 1",
            "height": 80,
            "id": rows[0]["id"],
            "image": str(first.resolve()),
            "license": "user-owned-local-evaluation",
            "sha256": rows[0]["sha256"],
            "source": "local-manhwa",
            "split": "stress",
            "style": "manhwa",
            "width": 120,
        }
    ]
    assert rows[0]["id"] == f"local-manhwa-{rows[0]['sha256'][:16]}"
