import json

import cv2

from tools.evaluate_masks import evaluate_manifest


def test_evaluator_reports_perfect_metrics_for_exact_prediction(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    from tools.generate_synthetic_dataset import generate_dataset

    generate_dataset(dataset, samples=1, seed=7)
    row = json.loads((dataset / "manifest.jsonl").read_text(encoding="utf-8").splitlines()[0])

    def exact_backend(image, bbox, text):
        return cv2.imread(str(dataset / row["text_mask"]), cv2.IMREAD_GRAYSCALE)

    report = evaluate_manifest(dataset / "manifest.jsonl", exact_backend, config_hash="a" * 64)

    assert report["summary"]["dice"] == 1.0
    assert report["summary"]["precision"] == 1.0
    assert report["config_hash"] == "a" * 64
