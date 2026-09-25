from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference_common import Detection  # noqa: E402

MODEL_PATH = Path(__file__).with_name("efficientdet_lite0.onnx")
INPUT_SIZE = 320


def preprocess(frame: np.ndarray) -> np.ndarray:
    image = cv2.cvtColor(cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE)), cv2.COLOR_BGR2RGB)
    return image[None].astype(np.uint8)


def decode(outputs: object, shape: tuple[int, ...], threshold: float) -> list[Detection]:
    boxes, class_ids, scores = outputs[0][0], outputs[1][0], outputs[2][0]
    height, width = shape[:2]
    detections = []
    for box, class_id, score in zip(boxes, class_ids, scores):
        score = float(score)
        class_id = int(class_id)
        if score < threshold or not 0 <= class_id < 80:
            continue
        ymin, xmin, ymax, xmax = np.clip(box, 0, 1) * INPUT_SIZE
        detections.append(Detection((int(xmin * width / INPUT_SIZE), int(ymin * height / INPUT_SIZE), int(xmax * width / INPUT_SIZE), int(ymax * height / INPUT_SIZE)), score, class_id))
    return detections


