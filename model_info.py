from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

try:
    import onnx
    from onnx import shape_inference
except ImportError as exc:
    raise SystemExit("Install benchmark dependencies: pip install -r requirements.txt") from exc


MODELS = [
    ("YOLOv5n", "yolo_inference/models/yolov5nu.onnx"),
    ("YOLOv8n", "yolo_inference/models/yolov8n.onnx"),
    ("YOLOv10n", "yolo_inference/models/yolov10n.onnx"),
    ("YOLOv11n", "yolo_inference/models/yolo11n.onnx"),
    ("YOLO26n", "yolo_inference/models/yolo26n.onnx"),
    ("MobileNetV2  SSD LITE", "ssd_mobilenet_inference/ssd_mobilenet_v2.onnx"),
    ("EfficientDet Lite0", "EfficientDet_inference/efficientdet_lite0.onnx"),
    ("LeYOLONano", "yolo_inference/models/LeYOLONano.onnx"),
    ("NanoDet-Plus", "NanoDet-Plus_inference/nanodet-plus-m-1.5x_320.onnx"),
    ("PicoDet-s", "PicoDet-s_inference/picodet_s_320_coco.onnx"),
]

KNOWN_RESOLUTIONS = {
    "ssd_mobilenet_v2.onnx": "300x300",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print ONNX model parameters and input details")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    return parser.parse_args()


def tensor_shapes(model: Any) -> dict[str, list[int | None]]:
    inferred = shape_inference.infer_shapes(model)
    shapes: dict[str, list[int | None]] = {}

    def record(value_info: Any) -> None:
        if not value_info.type.HasField("tensor_type"):
            return
        shapes[value_info.name] = [
            int(dim.dim_value) if dim.HasField("dim_value") else None
            for dim in value_info.type.tensor_type.shape.dim
        ]

    for item in (*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output):
        record(item)
    shapes.update({item.name: list(map(int, item.dims)) for item in inferred.graph.initializer})
    return shapes


def estimate_macs(model: Any) -> float | None:
    try:
        shapes = tensor_shapes(model)
    except Exception:
        return None

    total = 0.0
    found = False
    for node in model.graph.node:
        if len(node.input) < 2 or not node.output:
            continue
        output_shape = shapes.get(node.output[0])
        weight_shape = shapes.get(node.input[1])
        if node.op_type in {"Conv", "ConvTranspose"} and output_shape and weight_shape and len(weight_shape) >= 3:
            if any(d is None or d <= 0 for d in output_shape):
                continue
            total += math.prod(output_shape) * math.prod(weight_shape[2:]) * weight_shape[1]
            found = True
        elif node.op_type in {"Gemm", "MatMul"}:
            lhs, rhs = shapes.get(node.input[0]), shapes.get(node.input[1])
            if lhs and rhs and len(lhs) >= 2 and len(rhs) >= 2 and lhs[-1] and rhs[-1]:
                total += lhs[-2] * lhs[-1] * rhs[-1]
                found = True
    return total / 1e9 if found else None


def resolution(model: Any, path: Path) -> str:
    for value in model.graph.input:
        shape = value.type.tensor_type.shape.dim
        dimensions = [int(dim.dim_value) if dim.HasField("dim_value") else None for dim in shape]
        if len(dimensions) != 4 or dimensions[2] is None or dimensions[3] is None:
            continue
        if dimensions[1] in {1, 3}:
            return f"{dimensions[3]}x{dimensions[2]}"
        if dimensions[3] in {1, 3}:
            return f"{dimensions[2]}x{dimensions[1]}"
    return KNOWN_RESOLUTIONS.get(path.name, "-")


def fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    rows: list[list[str]] = []
    for name, relative_path in MODELS:
        path = root / relative_path
        if not path.exists():
            rows.append([name, "-", "-", "-", f"не найден: {relative_path}"])
            continue
        model = onnx.load(str(path), load_external_data=False)
        params_m = sum(math.prod(item.dims) for item in model.graph.initializer) / 1e6
        rows.append([
            name,
            fmt(params_m),
            fmt(estimate_macs(model)),
            f"{path.stat().st_size / 1048576:.2f}",
            resolution(model, path),
        ])

    headers = ["Модель", "Параметры, млн", "КОПТс", "Размер, МБ", "Разрешение"]
    widths = [max(len(header), *(len(row[index]) for row in rows)) for index, header in enumerate(headers)]
    border = "+" + "+".join("-" * (width + 2) for width in widths) + "+"
    print(border)
    print("|" + "|".join(f" {header.ljust(width)} " for header, width in zip(headers, widths)) + "|")
    print(border)
    for row in rows:
        print("|" + "|".join(f" {cell.ljust(width)} " for cell, width in zip(row[:len(headers)], widths)) + "|")
    print(border)
    for row in rows:
        if len(row) > len(headers):
            print(f"{row[0]}: {row[-1]}")


if __name__ == "__main__":
    main()