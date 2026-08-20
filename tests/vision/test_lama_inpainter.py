import numpy as np

from vision.inpainting.lama import (
    LamaCudaOutOfMemory,
    ResilientLamaInpainter,
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
