import json
from pathlib import Path

import cv2
import numpy as np

from tools.stress_test_comic_pages import run_manifest


def test_stress_runner_continues_after_page_error_and_hides_paths(tmp_path):
    image = np.full((120, 220, 3), 245, np.uint8)
    cv2.putText(
        image,
        "HELLO",
        (35, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    image_path = tmp_path / "page.jpg"
    assert cv2.imwrite(str(image_path), image)
    rows = [
        {
            "id": "good-page",
            "image": str(image_path),
            "style": "comic",
            "page_class": "story",
            "annotation_semantics": "detector-proposal",
            "blocks": [{"bbox": [25, 35, 145, 85], "score": 0.9}],
        },
        {
            "id": "missing-page",
            "image": str(tmp_path / "missing.jpg"),
            "style": "comic",
            "page_class": "story",
            "annotation_semantics": "detector-proposal",
            "blocks": [{"bbox": [0, 0, 20, 20], "score": 0.9}],
        },
    ]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    config = Path(__file__).resolve().parents[2] / "configs" / "vision.json"

    report = run_manifest(manifest, config, backend="heuristic", erase_pages=1)

    assert report["page_count"] == 2
    assert report["successful_pages"] == 1
    assert report["failed_pages"] == 1
    assert report["block_count"] == 1
    assert sum(report["decision_method_counts"].values()) == 1
    assert report["annotation_semantics"] == ["detector-proposal"]
    assert len(report["dataset_hash"]) == 64
    assert report["rows"][0]["id"] == "good-page"
    assert report["rows"][0]["outside_mask_delta"] == 0
    assert "image" not in report["rows"][0]
    assert report["rows"][1] == {
        "error": "could not decode image",
        "id": "missing-page",
        "status": "error",
        "style": "comic",
    }
