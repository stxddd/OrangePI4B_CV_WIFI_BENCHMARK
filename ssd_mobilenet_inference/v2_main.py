from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference_common import Detection, nms_indices  # noqa: E402

MODEL_PATH = Path(__file__).with_name("ssd_mobilenet_v2.onnx")
INPUT_SIZE = (300, 300)


def preprocess(frame: np.ndarray) -> np.ndarray:
    image = cv2.cvtColor(cv2.resize(frame, INPUT_SIZE), cv2.COLOR_BGR2RGB)
    return image[None].astype(np.uint8)


def decode(outputs: object, shape: tuple[int, ...], threshold: float) -> list[Detection]:
    boxes, classes, scores = outputs[1][0], outputs[2][0], outputs[4][0]
    count = min(int(outputs[5][0]), len(boxes))
    height, width = shape[:2]
    detections = []
    for box, class_id, score in zip(boxes[:count], classes[:count], scores[:count]):
        score = float(score)
        if score < threshold:
            continue
        detections.append(Detection((int(box[1] * width), int(box[0] * height), int(box[3] * width), int(box[2] * height)), score, int(class_id) - 1))
    keep = nms_indices([detection.box for detection in detections], [detection.score for detection in detections], threshold, 0.45)
    return [detections[index] for index in keep]


