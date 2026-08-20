import numpy as np

from vision.inpainting.lama import (
    build_lama_inpainter,
    discover_lama_checkpoint,
    LamaCudaOutOfMemory,
    ResilientLamaInpainter,
    TorchLamaBackend,
)


class RecordingBackend:
    def __init__(self, failures=0):
        self.failures = failures
        self.calls = []
        self.clear_calls = 0

    def inpaint(self, image, mask):
        self.calls.append((image.copy(), mask.copy()))
        if len(self.calls) <= self.failures:
            raise LamaCudaOutOfMemory("synthetic CUDA OOM")
        result = image.copy()
        result[:] = (10, 20, 240)
        return result

    def clear_cuda_cache(self):
        self.clear_calls += 1


def _page_and_mask():
    page = np.full((240, 320, 3), 80, np.uint8)
    mask = np.zeros((240, 320), np.uint8)
    mask[105:125, 145:175] = 255
    return page, mask


def test_lama_uses_one_full_resolution_call_and_composites_only_mask():
    page, mask = _page_and_mask()
    backend = RecordingBackend()
    lama = ResilientLamaInpainter(backend, context_min_px=32)

    output = lama.inpaint(page, mask)

    assert len(backend.calls) == 1
    assert backend.calls[0][0].shape == page.shape
    assert lama.last_run.mode == "full_page"
    assert np.all(output[mask > 0] == (10, 20, 240))
    assert np.array_equal(output[mask == 0], page[mask == 0])


def test_cuda_oom_retries_with_context_crop_at_original_resolution():
    page, mask = _page_and_mask()
    backend = RecordingBackend(failures=1)
    lama = ResilientLamaInpainter(
        backend,
        context_min_px=32,
        context_max_mask_ratio=0.08,
    )

    output = lama.inpaint(page, mask)

    assert len(backend.calls) == 2
    assert backend.calls[0][0].shape == page.shape
    retry_image, retry_mask = backend.calls[1]
    assert retry_image.shape[0] < page.shape[0]
    assert retry_image.shape[1] < page.shape[1]
    assert retry_image.shape[:2] == retry_mask.shape
    assert np.count_nonzero(retry_mask) == np.count_nonzero(mask)
    assert np.count_nonzero(retry_mask) / retry_mask.size <= 0.081
    assert retry_image.shape[0] >= 20 + 2 * 32
    assert retry_image.shape[1] >= 30 + 2 * 32
    assert backend.clear_calls == 1
    assert lama.last_run.mode == "context_crop"
    assert np.array_equal(output[mask == 0], page[mask == 0])


def test_second_cuda_oom_falls_back_to_telea_and_reports_warning():
    page, mask = _page_and_mask()
    backend = RecordingBackend(failures=2)
    lama = ResilientLamaInpainter(backend, context_min_px=32)

    output = lama.inpaint(page, mask)

    assert len(backend.calls) == 2
    assert backend.clear_calls == 2
    assert output.shape == page.shape
    assert lama.last_run.mode == "telea_fallback"
    assert "CUDA OOM" in lama.last_run.warning
    assert np.array_equal(output[mask == 0], page[mask == 0])


def test_torch_backend_keeps_native_size_and_pads_model_input_to_eight():
    import torch

    class FakeModel:
        def __init__(self):
            self.calls = []

        def eval(self):
            return self

        def __call__(self, model_input):
            self.calls.append(model_input.detach().cpu())
            batch, _, height, width = model_input.shape
            return torch.full(
                (batch, 3, height, width),
                0.25,
                device=model_input.device,
            )

    image = np.full((73, 119, 3), (10, 20, 30), np.uint8)
    mask = np.zeros((73, 119), np.uint8)
    mask[20:35, 40:70] = 255
    model = FakeModel()
    backend = TorchLamaBackend(model=model, device="cpu", precision="fp32")

    output = backend.inpaint(image, mask)

    assert output.shape == image.shape
    assert output.dtype == np.uint8
    assert len(model.calls) == 1
    model_input = model.calls[0]
    assert tuple(model_input.shape) == (1, 4, 80, 120)
    assert torch.all(model_input[0, :3, 25, 50] == 0)
    assert model_input[0, 3, 25, 50] == 1
    assert torch.all(model_input[0, 3, 73:, :] == 0)
    assert torch.all(model_input[0, 3, :, 119:] == 0)
    assert np.all(output == 64)
    assert backend.last_elapsed_ms > 0


def test_torch_backend_normalizes_cuda_out_of_memory():
    import pytest
    import torch

    class OomModel:
        def eval(self):
            return self

        def __call__(self, model_input):
            raise torch.cuda.OutOfMemoryError("synthetic allocation failure")

    backend = TorchLamaBackend(model=OomModel(), device="cpu", precision="fp32")

    with pytest.raises(LamaCudaOutOfMemory, match="synthetic allocation failure"):
        backend.inpaint(
            np.zeros((16, 16, 3), np.uint8),
            np.full((16, 16), 255, np.uint8),
        )


def test_torch_backend_retries_fp32_when_fp16_output_is_not_finite(monkeypatch):
    import torch

    class UnstableHalfModel:
        def __init__(self):
            self.calls = 0

        def eval(self):
            return self

        def __call__(self, model_input):
            self.calls += 1
            batch, _, height, width = model_input.shape
            value = float("nan") if self.calls == 1 else 0.5
            return torch.full(
                (batch, 3, height, width), value, device=model_input.device
            )

    model = UnstableHalfModel()
    backend = TorchLamaBackend(model=model, device="cpu", precision="fp16")
    monkeypatch.setattr(backend, "_use_fp16", lambda device: True)

    output = backend.inpaint(
        np.zeros((16, 16, 3), np.uint8),
        np.full((16, 16), 255, np.uint8),
    )

    assert model.calls == 2
    assert np.all(output == 128)
    assert backend.last_precision == "fp32_fallback"

    backend.inpaint(
        np.zeros((16, 16, 3), np.uint8),
        np.full((16, 16), 255, np.uint8),
    )

    assert model.calls == 3
    assert backend.last_precision == "fp32"


def test_checkpoint_discovery_prefers_explicit_then_environment(tmp_path, monkeypatch):
    explicit = tmp_path / "explicit.ckpt"
    environment = tmp_path / "environment.ckpt"
    explicit.write_bytes(b"explicit")
    environment.write_bytes(b"environment")
    monkeypatch.setenv("LAMA_CHECKPOINT", str(environment))

    assert discover_lama_checkpoint(explicit) == explicit.resolve()
    assert discover_lama_checkpoint() == environment.resolve()


def test_torch_backend_retries_fp32_after_cufft_half_precision_error(monkeypatch):
    import torch

    class CufftModel:
        def __init__(self):
            self.calls = 0

        def eval(self):
            return self

        def __call__(self, model_input):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError(
                    "cuFFT only supports dimensions whose sizes are powers of two "
                    "when computing in half precision"
                )
            batch, _, height, width = model_input.shape
            return torch.full(
                (batch, 3, height, width), 0.5, device=model_input.device
            )

    model = CufftModel()
    backend = TorchLamaBackend(model=model, device="cpu", precision="fp16")
    monkeypatch.setattr(backend, "_use_fp16", lambda device: True)

    output = backend.inpaint(
        np.zeros((16, 24, 3), np.uint8),
        np.full((16, 24), 255, np.uint8),
    )

    assert model.calls == 2
    assert np.all(output == 128)
    assert backend.last_precision == "fp32_fallback"


def test_build_lama_inpainter_wires_cuda_and_context_without_loading_model(tmp_path):
    checkpoint = tmp_path / "lama.ckpt"
    checkpoint.write_bytes(b"lazy checkpoint placeholder")

    lama = build_lama_inpainter(
        checkpoint,
        device="cuda",
        precision="fp16",
        context_min_px=123,
        context_max_mask_ratio=0.07,
        telea_radius=5,
    )

    assert isinstance(lama, ResilientLamaInpainter)
    assert isinstance(lama.backend, TorchLamaBackend)
    assert lama.backend.checkpoint_path == checkpoint.resolve()
    assert lama.backend.device_name == "cuda"
    assert lama.context_min_px == 123
    assert lama.context_max_mask_ratio == 0.07
    assert lama.telea_radius == 5
