from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

COCO = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
]


@dataclass(slots=True)
class Detection:
    box: tuple[int, int, int, int]
    score: float
    class_id: int


def create_session(model_path: Path, threads: int = 4):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = max(1, min(threads, 4))
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(
        str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
    )


def nms_indices(
    boxes: Sequence[Sequence[int]], scores: Sequence[float],
    score_threshold: float, nms_threshold: float,
) -> list[int]:
    if not boxes:
        return []
    rects = [[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2 in boxes]
    ids = cv2.dnn.NMSBoxes(rects, list(scores), score_threshold, nms_threshold)
    return np.asarray(ids).reshape(-1).astype(int).tolist() if len(ids) else []


def draw_detections(frame: np.ndarray, detections: Sequence[Detection]) -> None:
    height, width = frame.shape[:2]
    for detection in detections:
        x1, y1, x2, y2 = detection.box
        x1, x2 = max(0, min(x1, width - 1)), max(0, min(x2, width - 1))
        y1, y2 = max(0, min(y1, height - 1)), max(0, min(y2, height - 1))
        label = COCO[detection.class_id] if 0 <= detection.class_id < len(COCO) else str(detection.class_id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.putText(
            frame, f"{label} {detection.score:.2f}", (x1, max(16, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 2, cv2.LINE_AA,
        )


