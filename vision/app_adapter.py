"""Default-off bridge from the Flask renderer to VisionPipeline."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
from typing import Callable, Iterable, Mapping

import numpy as np

from vision.config import VisionConfig
from vision.pipeline import VisionPipeline
from vision.types import EraseResult, PreparedBlock


_LOGGER = logging.getLogger(__name__)
_TRUE_VALUES = {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class VisionPageExecution:
    prepared: list[PreparedBlock]
    erased_image: np.ndarray
    erase_results: list[EraseResult]


class VisionPageAdapter:
    """Prepare and erase an image once without persisting runtime arrays."""

    def __init__(
        self,
        pipeline: VisionPipeline | object | None = None,
        config: VisionConfig | None = None,
    ) -> None:
        self.config = config or _load_default_config()
        self.pipeline = pipeline or VisionPipeline(config=self.config)

    def prepare_page(
        self,
        image: np.ndarray,
        blocks: Iterable[Mapping[str, object]],
    ) -> list[PreparedBlock]:
        return list(self.pipeline.prepare_page(image, blocks))

    def erase_page(
        self,
        image: np.ndarray,
        prepared: list[PreparedBlock],
    ) -> tuple[np.ndarray, list[EraseResult]]:
        erased, results = self.pipeline.erase_page(image, prepared)
        return erased, list(results)

    def process_page(
        self,
        image: np.ndarray,
        blocks: Iterable[Mapping[str, object]],
    ) -> VisionPageExecution:
        prepared = self.prepare_page(image, blocks)
        erased_image, erase_results = self.erase_page(image, prepared)
        return VisionPageExecution(
            prepared=prepared,
            erased_image=erased_image,
            erase_results=erase_results,
        )


def build_optional_vision_adapter(
    *,
    pipeline_factory: Callable[..., object] = VisionPipeline,
) -> VisionPageAdapter | None:
    """Create the adapter only when explicitly enabled by the environment."""
    enabled = os.environ.get("MANGA_VISION_PIPELINE", "").strip().lower()
    if enabled not in _TRUE_VALUES:
        return None
    config = _load_default_config()
    try:
        pipeline = pipeline_factory(config=config)
        return VisionPageAdapter(pipeline=pipeline, config=config)
    except Exception as exc:
        _LOGGER.warning("Vision pipeline disabled after initialization failure: %s", exc)
        return None


def _load_default_config() -> VisionConfig:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "vision.json"
    return VisionConfig.load(config_path)
