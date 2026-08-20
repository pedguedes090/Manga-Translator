"""Validated configuration for the vision pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any


_DEFAULTS: dict[str, Any] = {
    "profile": "cuda",
    "mask_backend": "auto",
    "allow_cpu_fallback": True,
    "text_mask": {
        "input_size": 512,
        "crop_padding_ratio": 0.12,
        "prob_high": 0.62,
        "prob_low": 0.34,
        "max_coverage": 0.65,
        "max_bubble_border_overlap": 0.02,
        "dilation_min_px": 1,
        "dilation_max_px": 4,
    },
    "bubble": {
        "enabled": True,
        "match_confidence": 0.45,
        "safe_border_px": 3,
    },
    "inpaint": {
        "strategy": "auto",
        "flat_max_texture_std": 12.0,
        "telea_radius": 3,
        "lama_full_resolution": True,
        "precision": "fp16",
        "oom_context_min_px": 256,
        "oom_context_max_mask_ratio": 0.08,
    },
    "safety": {"manual_review_on_uncertain": True},
    "debug": {"save_artifacts": False},
}


@dataclass(frozen=True)
class TextMaskConfig:
    input_size: int
    crop_padding_ratio: float
    prob_high: float
    prob_low: float
    max_coverage: float
    max_bubble_border_overlap: float
    dilation_min_px: int
    dilation_max_px: int


@dataclass(frozen=True)
class BubbleConfig:
    enabled: bool
    match_confidence: float
    safe_border_px: int


@dataclass(frozen=True)
class InpaintConfig:
    strategy: str
    flat_max_texture_std: float
    telea_radius: int
    lama_full_resolution: bool
    precision: str
    oom_context_min_px: int
    oom_context_max_mask_ratio: float


@dataclass(frozen=True)
class SafetyConfig:
    manual_review_on_uncertain: bool


@dataclass(frozen=True)
class DebugConfig:
    save_artifacts: bool


@dataclass(frozen=True)
class VisionConfig:
    profile: str
    mask_backend: str
    allow_cpu_fallback: bool
    text_mask: TextMaskConfig
    bubble: BubbleConfig
    inpaint: InpaintConfig
    safety: SafetyConfig
    debug: DebugConfig

    @classmethod
    def load(cls, path: str | Path) -> VisionConfig:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("vision config must be a JSON object")

        unknown = sorted(set(raw) - set(_DEFAULTS))
        if unknown:
            raise ValueError(f"unknown vision config keys: {', '.join(unknown)}")

        data = _merge_defaults(_DEFAULTS, raw)
        config = cls(
            profile=str(data["profile"]),
            mask_backend=str(data["mask_backend"]),
            allow_cpu_fallback=bool(data["allow_cpu_fallback"]),
            text_mask=TextMaskConfig(**data["text_mask"]),
            bubble=BubbleConfig(**data["bubble"]),
            inpaint=InpaintConfig(**data["inpaint"]),
            safety=SafetyConfig(**data["safety"]),
            debug=DebugConfig(**data["debug"]),
        )
        config._validate()
        return config

    def config_hash(self) -> str:
        canonical = json.dumps(
            asdict(self), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        return sha256(canonical).hexdigest()

    def _validate(self) -> None:
        text_mask = self.text_mask
        if not 0 <= text_mask.prob_low < text_mask.prob_high <= 1:
            raise ValueError("prob_low must be less than prob_high")
        if text_mask.input_size <= 0:
            raise ValueError("text mask input_size must be positive")
        if not 0 < text_mask.max_coverage <= 1:
            raise ValueError("max_coverage must be in (0, 1]")
        if not 0 <= text_mask.max_bubble_border_overlap <= 1:
            raise ValueError("max_bubble_border_overlap must be in [0, 1]")
        if text_mask.dilation_min_px > text_mask.dilation_max_px:
            raise ValueError("dilation_min_px must not exceed dilation_max_px")


def _merge_defaults(defaults: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for key, default_value in defaults.items():
        override_value = overrides.get(key, default_value)
        if isinstance(default_value, dict):
            if not isinstance(override_value, dict):
                raise ValueError(f"{key} must be a JSON object")
            unknown = sorted(set(override_value) - set(default_value))
            if unknown:
                raise ValueError(f"unknown {key} keys: {', '.join(unknown)}")
            merged[key] = _merge_defaults(default_value, override_value)
        else:
            merged[key] = override_value
    return merged
