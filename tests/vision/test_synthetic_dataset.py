import json

from tools.generate_synthetic_dataset import generate_dataset


def test_synthetic_dataset_is_deterministic_and_uses_stable_names(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"

    generate_dataset(first, samples=3, seed=20260820)
    generate_dataset(second, samples=3, seed=20260820)

    first_rows = [json.loads(line) for line in (first / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    second_rows = [json.loads(line) for line in (second / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]

    assert len(first_rows) == 3
    assert first_rows == second_rows
    assert first_rows[0]["image"] == "image_0000.png"
    assert first_rows[0]["text_mask"] == "text_mask_0000.png"
    assert (first / first_rows[0]["image"]).read_bytes() == (second / second_rows[0]["image"]).read_bytes()
