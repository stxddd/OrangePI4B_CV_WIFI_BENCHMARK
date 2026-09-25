from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference_dfl import decode_single, preprocess  # noqa: E402

MODEL_PATH = Path(__file__).with_name("nanodet-plus-m-1.5x_320.onnx")


