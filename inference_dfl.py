from __future__ import annotations

import cv2
import numpy as np

from inference_common import Detection, nms_indices

INPUT_SIZE = 320
REG_MAX = 7
POINTS = np.asarray(
    [[x * stride, y * stride, stride]
     for stride in (8, 16, 32, 64)
     for y in range(INPUT_SIZE // stride)
     for x in range(INPUT_SIZE // stride)],
    dtype=np.float32,
)
MEAN = np.asarray([103.53, 116.28, 123.675], dtype=np.float32)
STD = np.asarray([57.375, 57.12, 58.395], dtype=np.float32)


def preprocess(frame: np.ndarray) -> np.ndarray:
    image = cv2.cvtColor(cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE)), cv2.COLOR_BGR2RGB).astype(np.float32)
    return np.ascontiguousarray(((image - MEAN) / STD).transpose(2, 0, 1)[None])


def decode_single(pred: np.ndarray, shape: tuple[int, ...], threshold: float) -> list[Detection]:
    if isinstance(pred, (list, tuple)):
        pred = pred[0]
    scores_by_class, regression = pred[0, :, :80], pred[0, :, 80:]
    labels = np.argmax(scores_by_class, axis=1)
    scores = np.max(scores_by_class, axis=1)
    return decode_arrays(scores, labels, regression, shape, threshold)


def decode_pico(outputs: object, shape: tuple[int, ...], threshold: float) -> list[Detection]:
    scores = np.concatenate([output[0] for output in outputs[:4]]).reshape(-1, 80)
    regression = np.concatenate([output[0] for output in outputs[4:]])
    labels = np.argmax(scores, axis=1)
    return decode_arrays(np.max(scores, axis=1), labels, regression, shape, threshold)


def decode_arrays(scores: np.ndarray, labels: np.ndarray, regression: np.ndarray, shape: tuple[int, ...], threshold: float) -> list[Detection]:
    mask = scores >= threshold
    if not np.any(mask):
        return []
    points, regression, scores, labels = POINTS[mask], regression[mask], scores[mask], labels[mask]
    values = regression.reshape(-1, REG_MAX + 1)
    values -= values.max(axis=1, keepdims=True)
    values = np.exp(values)
    values /= values.sum(axis=1, keepdims=True)
    distances = (values * np.arange(REG_MAX + 1, dtype=np.float32)).sum(axis=1).reshape(-1, 4) * points[:, 2, None]
    boxes = np.column_stack((points[:, 0] - distances[:, 0], points[:, 1] - distances[:, 1], points[:, 0] + distances[:, 2], points[:, 1] + distances[:, 3]))
    height, width = shape[:2]
    boxes[:, [0, 2]] *= width / INPUT_SIZE
    boxes[:, [1, 3]] *= height / INPUT_SIZE
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height - 1)
    keep = nms_indices([[int(x) for x in box] for box in boxes], scores, threshold, 0.5)
    return [Detection(tuple(boxes[i].astype(int)), float(scores[i]), int(labels[i])) for i in keep]
