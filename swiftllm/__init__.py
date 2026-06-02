from __future__ import annotations

from pathlib import Path

# Local-development compatibility:
# Allow `import swiftllm` from repo root without `pip install -e .`
# by extending package search path to `src/swiftllm`.
_ROOT = Path(__file__).resolve().parents[1]
_SRC_PKG = _ROOT / "src" / "swiftllm"
if _SRC_PKG.exists():
    __path__.append(str(_SRC_PKG))  # type: ignore[name-defined]

from .config import BenchmarkConfig, Config, DataConfig, ModelConfig, TargetConfig, TrainConfig, load_config
from .model import SwiftLLM
from .train_loop import train

__all__ = [
    "Config",
    "DataConfig",
    "ModelConfig",
    "TrainConfig",
    "TargetConfig",
    "BenchmarkConfig",
    "load_config",
    "SwiftLLM",
    "train",
]

