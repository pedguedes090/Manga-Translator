# Full vision pipeline execution roadmap

This roadmap implements [the approved full vision pipeline design](../specs/2026-08-20-full-vision-pipeline-design.md) as four independently testable plans. Execute them in order because each later plan consumes interfaces produced by the previous one.

| Order | Detailed plan | Working deliverable |
| ---: | --- | --- |
| 1 | [Vision foundation and hybrid masking](2026-08-20-vision-foundation-and-hybrid.md) | Frozen baseline, typed vision results, extracted region analysis, heuristic compatibility, hybrid masker, and one-mask preparation pipeline. |
| 2 | [CUDA model runtime and segmentation](2026-08-20-cuda-model-runtime-and-segmentation.md) | Verified model downloads, explicit CUDA provider selection, neural text masks, bubble masks, reproducible training, and ONNX export. |
| 3 | [Inpainting and application integration](2026-08-20-inpainting-and-application-integration.md) | Flat/Telea/LaMa routing, full-resolution page inference, OOM fallback, session mask cache, Flask integration, and CLI options. |
| 4 | [Evaluation and rollout](2026-08-20-vision-evaluation-and-rollout.md) | Curated benchmarks, threshold calibration, end-to-end reports, CUDA documentation, and evidence-gated legacy deprecation. |

## Delivery protocol

Every task in every detailed plan follows this order:

1. `git fetch origin` and compare `main...origin/main`.
2. If the remote is ahead, rebase without force-pushing and preserve unrelated working-tree changes.
3. Write the failing test before production code.
4. Run the focused test and observe the expected failure.
5. Implement the smallest complete behavior for that task.
6. Run the focused test and `translator/test_translator.py`.
7. Stage only the paths listed in the task.
8. Commit once and push directly to `origin/main`.

If a push is rejected, fetch, rebase, rerun the affected tests, and push normally. Never stage the existing unrelated changes under `.commandcode/`, `.jules/`, or `__pycache__/`.

## Rollout gates

- A plan starts only after the preceding plan's complete test suite passes on `main`.
- Hybrid does not become default until it improves hard-mask Dice by at least 15 percent over the frozen heuristic baseline.
- Neural text masking does not become default until it beats hybrid Dice without reducing artwork-preservation precision.
- Bubble gating does not become mandatory until bubble-border damage remains below one percent.
- Full-resolution LaMa remains restricted to complex masks and must preserve every outside-mask pixel exactly.
- Legacy helpers are deprecated only in the final plan and only after all gates above pass.
