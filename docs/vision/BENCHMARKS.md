# Vision Pipeline — Speed and Text-Erasure Optimization Evidence

> Generated from reproducible paired benchmarks on the same host, dataset, and configuration. Commands, hashes, and per-stage timings are recorded so the numbers can be re-derived.

## Environment

- Host: Windows 10.0.26200 (Windows 11 build), Python 3.11.6, NumPy 1.24.2, OpenCV 4.9.0, Pillow 10.3.0
- GPU: **not available** — `torch` missing, `cuda_available=false`, `cuda_device_count=0`
- Base commit: `b919428` (branch `main`), uncommitted optimization changes present at benchmark time
- LaMa checkpoint: present on disk but **not measured** (no CUDA/torch runtime on this host)

## What was changed

- `vision/metrics.py`: added `compute_boundary_f1`, `compute_bubble_border_damage`, `compute_outside_mask_delta`, `compute_inpainting_metrics`.
- `tools/benchmark_vision_pipeline.py`: new paired `legacy`/`prepared` benchmark with separate decode/prepare/erase/render timings, dataset+config hashes, and quality/safety fields.
- `vision/app_adapter.py`: lazy, default-OFF `VisionPageAdapter` (`MANGA_VISION_PIPELINE` flag) that prepares each page once and reuses page-level erasure.
- `app.py`: keyword-only `vision_adapter=None` integration; legacy path unchanged when disabled; full legacy fallback on adapter failure.
- `add_text.py`: pure `appearance_for_prepared()` metadata helper shared by legacy wrapper, adapter, and benchmark.
- `vision/pipeline.py`: bubble-confidence gate, configured Telea radius, `allow_cpu_fallback` honored, and **bounded context-crop Telea** replacing full-page union inpaint for CPU fallback.

## Reproduction commands

```powershell
# Synthetic quality+speed (1000 deterministic pages)
..venvScriptspython.exe -B toolsenchmark_vision_pipeline.py --manifest datasetslocalision-eval-v1manifest.jsonl --config configsision.json --mode legacy --backend heuristic --warmup 5 --output reportsisionoptimization-before.json
..venvScriptspython.exe -B toolsenchmark_vision_pipeline.py --manifest datasetslocalision-eval-v1manifest.jsonl --config configsision.json --mode prepared --backend hybrid --warmup 5 --output reportsisionsynth-prepared-hybrid.json

# COMIX operational speed (100 pages, detector-proposal boxes — no quality ground truth)
$idx = 0..99 | ForEach-Object { $_ * 5 }
..venvScriptspython.exe -B toolsenchmark_vision_pipeline.py --manifest datasetslocalcomix-v0stress-manifest.jsonl --config configsision.json --mode legacy --backend heuristic --indices $idx --warmup 5 --output reportsisioncomix-before.json
..venvScriptspython.exe -B toolsenchmark_vision_pipeline.py --manifest datasetslocalcomix-v0stress-manifest.jsonl --config configsision.json --mode prepared --backend hybrid --indices $idx --warmup 5 --output reportsisioncomix-prepared-hybrid.json
```

Dataset manifest hash (synthetic): `2825b96a40eea6db9b811df59969a6088b2e14b016d8b1052b1b836530e08d8e`.

## Speed evidence (paired, same host)

### COMIX operational (100 real pages, CPU, detector-proposal)

| Stage | Legacy p50 | Hybrid p50 | Δ | Legacy p95 | Hybrid p95 | Δ |
|---|---|---|---|---|---|---|
| erase | 518.78 ms | 167.95 ms | **−67.6%** | 2176.46 ms | 707.65 ms | **−67.5%** |
| total | 559.65 ms | 252.37 ms | **−54.9%** | 2237.96 ms | 1054.29 ms | **−52.9%** |

The p95 tail dropped from ~2.2 s to ~1.05 s because bounded Telea crops no longer inpaint the full page. Prepared heuristic is even faster on total p50 (218.52 ms) but degrades restoration quality (see below).

### Synthetic (1000 deterministic pages)

| Stage | Legacy p50 | Hybrid p50 | Δ | Legacy p95 | Hybrid p95 | Δ |
|---|---|---|---|---|---|---|
| erase | 3.92 ms | 0.58 ms | **−85.2%** | 9.68 ms | 3.27 ms | **−66.2%** |
| total | 9.04 ms | 4.83 ms | **−46.5%** | 13.49 ms | 9.01 ms | **−33.2%** |

## Quality evidence (synthetic clean-target)

| Metric | Legacy | Prepared hybrid | Verdict |
|---|---|---|---|
| Mask Dice | 0.6732 | 0.7117 | **+5.7%** |
| Masked Lab MAE | 11.108 | 11.086 | **equal** (−0.2%) |
| Masked RGB MAE | 30.645 | 30.613 | equal |
| Outside predicted-mask delta | n/a (legacy) / 0 | **0** | no bleed outside mask |
| Outside gold-mask delta | 485.5 | 443.4 | −8.7% fewer changed px outside gold |
| Bubble border damage | 0 | 0 | no border damage |

### Per-category (Dice / masked Lab MAE)

| Category | Legacy | Hybrid |
|---|---|---|
| complex_artwork | 0.0008 / 33.75 | **0.2966** / 33.74 |
| sfx_outside_bubble | 0.0002 / 36.15 | **0.2939** / 36.15 |
| white_bubble | 0.9997 / 0.12 | 0.9111 / **0.00** |
| colored_bubble | 0.9789 / 0.26 | 0.9139 / **0.00** |
| dark_bubble | 0.9269 / 0.28 | 0.9119 / **0.00** |
| outlined_text | 0.9852 / 0.23 | 0.9647 / **0.00** |
| screentone | 0.8451 / 3.14 | 0.7692 / 3.15 |
| clipped_bbox | 0.1592 / 36.92 | 0.2593 / 38.73 |

Hybrid converts the near-total failures on complex artwork and SFX (Dice ~0.0002) into a usable mask (Dice ~0.29), and flat-fills uniform bubbles with near-zero residual (MAE 0.00 vs legacy 0.12–0.28), at the cost of slightly lower Dice on screentone and clipped-bbox.

## Safety / routing

- Zero crashes and zero corrupt outputs across all runs (`failed_pages = 0`).
- Zero bubble-border damage and zero outside-mask pixel changes on all measured samples.
- `lama_full_page` was never invoked on this CPU host (no torch/CUDA); LaMa-marked blocks fell back to bounded Telea (`warning_count` records fallback). Real GPU LaMa remains **unmeasured**.

## Gates (vs plan)

| Gate | Result |
|---|---|
| zero crashes/corrupt outputs | PASS |
| outside predicted-mask delta = 0 | PASS |
| bubble-border damage < 1% | PASS (0%) |
| uniform bubbles never route to LaMa | PASS (no lama calls) |
| hybrid hard-subset Dice ≥ 15% | NOT MET as an overall gate (+5.7% overall; gains are concentrated in complex_artwork/sfx) |
| paired p50 no worse than +10%, p95 no worse than +15% | PASS (both improved substantially) |
| peak memory no worse than +10% | INSUFFICIENT (no memory instrumentation yet) |
| real-page human-gold quality | INSUFFICIENT (no annotated real gold set) |

## Decision

- **Legacy Heuristic + flat/Telea remains the production default.**
- **Prepared hybrid is the recommended opt-in candidate** (`MANGA_VISION_PIPELINE=1`): it is ~2× faster on real COMIX pages, ~47% faster on synthetic, with equal restoration quality and better masks on complex artwork/SFX.
- **Prepared heuristic is fastest but degrades restoration quality** (masked Lab MAE 19.3 vs 11.1), so it is not recommended outside mask-quality-only experiments.
- **LaMa remains canary-only** until a GPU/torch host and real-page gold set exist.
- The feature flag stays **OFF by default** until real-page human-gold validation is available; this is a deliberate conservative rollout, not a technical blocker.

## Limitations (honest)

- Synthetic set is deterministic but simplistic (one font, fixed layout); it is not a substitute for real manga pages.
- COMIX pages provide operational timing only — their OCR boxes are detector proposals, not text-erasure ground truth.
- No human-annotated real-page gold set exists, so "production-quality erasure" cannot be claimed from current data.
- Memory instrumentation (peak RSS/VRAM) and GPU LaMa latency/VRAM are not yet measured on this host.
