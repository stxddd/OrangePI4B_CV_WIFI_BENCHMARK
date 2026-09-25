from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference_common import Detection, nms_indices  # noqa: E402

MODEL_DIR = Path(__file__).with_name("models")
MODELS = {
    "yolo5n": "yolov5nu.onnx",
    "yolo8n": "yolov8n.onnx",
    "yolo10n": "yolov10n.onnx",
    "yolo11n": "yolo11n.onnx",
    "yolo26n": "yolo26n.onnx",
    "leyolo_nano": "LeYOLONano.onnx",
}
INPUT_SIZE = 320


def letterbox(frame: np.ndarray) -> tuple[np.ndarray, float, int, int]:
    height, width = frame.shape[:2]
    scale = min(INPUT_SIZE / width, INPUT_SIZE / height)
    resized = cv2.resize(frame, (int(width * scale), int(height * scale)))
    canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
    dx, dy = (INPUT_SIZE - resized.shape[1]) // 2, (INPUT_SIZE - resized.shape[0]) // 2
    canvas[dy:dy + resized.shape[0], dx:dx + resized.shape[1]] = resized
    return canvas, scale, dx, dy


def preprocess(frame: np.ndarray) -> np.ndarray:
    image, _, _, _ = letterbox(frame)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.ascontiguousarray(image.transpose(2, 0, 1)[None])


def decode(outputs: object, shape: tuple[int, ...], threshold: float) -> list[Detection]:
    output = np.squeeze(outputs[0])
    if output.shape[-1] == 6:
        rows = output
    else:
        if output.shape[0] in (84, 85):
            output = output.T
        rows = []
        for row in output:
            objectness = row[4] if len(row) == 85 else 1.0
            scores = row[5:] * objectness if len(row) == 85 else row[4:]
            class_id = int(np.argmax(scores))
            score = float(scores[class_id])
            if score >= threshold:
                x, y, width, height = row[:4]
                rows.append([x - width / 2, y - height / 2, x + width / 2, y + height / 2, score, class_id])
        rows = np.asarray(rows, dtype=np.float32)
    if len(rows) == 0:
        return []
    frame_height, frame_width = shape[:2]
    frame_height, frame_width = shape[:2]
    scale = min(INPUT_SIZE / frame_width, INPUT_SIZE / frame_height)
    resized_width, resized_height = int(frame_width * scale), int(frame_height * scale)
    dx, dy = (INPUT_SIZE - resized_width) // 2, (INPUT_SIZE - resized_height) // 2
    boxes, scores, class_ids = [], [], []
    for x1, y1, x2, y2, score, class_id in rows:
        x1, x2 = (x1 - dx) / scale, (x2 - dx) / scale
        y1, y2 = (y1 - dy) / scale, (y2 - dy) / scale
        boxes.append([max(0, int(x1)), max(0, int(y1)), min(frame_width - 1, int(x2)), min(frame_height - 1, int(y2))])
        scores.append(float(score))
        class_ids.append(int(class_id))
    keep = nms_indices(boxes, scores, threshold, 0.45)
    return [Detection(tuple(boxes[i]), scores[i], class_ids[i]) for i in keep]


