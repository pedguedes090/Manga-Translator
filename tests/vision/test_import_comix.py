import json
from pathlib import Path

import numpy as np
from PIL import Image

from tools.import_comix_stress import import_comix


def test_import_comix_clamps_and_filters_detector_boxes(tmp_path):
    source = tmp_path / "comix"
    source.mkdir()
    image_path = source / "c00001_p002.jpg"
    Image.fromarray(np.full((100, 200, 3), 220, np.uint8)).save(image_path)
    annotation = {
        "page_id": "c00001_p002",
        "book_id": "c00001",
        "page_number": 2,
        "page_class": "story",
        "image": {"file": image_path.name, "width": 200, "height": 100},
        "detections": {
            "magi": {
                "text": [
                    {"bbox": [-5.2, 10.1, 40.8, 50.6], "score": 0.8},
                    {"bbox": [60, 20, 80, 40], "score": 0.1},
                ]
            }
        },
    }
    (source / "c00001_p002.json").write_text(
        json.dumps(annotation), encoding="utf-8"
    )
    output = tmp_path / "manifest.jsonl"

    summary = import_comix(source, output, score_threshold=0.25)

    row = json.loads(output.read_text(encoding="utf-8"))
    assert summary == {
        "block_count": 1,
        "invalid_page_count": 0,
        "page_count": 1,
        "pages_without_text": 0,
    }
    assert row["blocks"] == [
        {"bbox": [0, 10, 41, 51], "detector": "magi", "score": 0.8}
    ]
    assert row["annotation_semantics"] == "detector-proposal"
    assert row["license"] == "CC0-1.0"
    assert row["style"] == "comic"
    assert row["image"] == str(image_path.resolve())
