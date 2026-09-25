from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import cv2

from inference_common import create_session, draw_detections
from wifi_source import load_wifi_frames

try:
    import onnx
    from onnx import shape_inference
except ImportError as exc:
    raise SystemExit("Install benchmark dependencies: pip install -r requirements.txt") from exc


@dataclass
class ModelRow:
    name: str
    path: Path
    params_m: float | None = None
    gmacs: float | None = None
    size_mb: float = 0.0
    frames: int = 0
    avg_fps: float | None = None
    min_fps: float | None = None
    max_fps: float | None = None
    avg_latency_ms: float | None = None
    min_latency_ms: float | None = None
    max_latency_ms: float | None = None
    video_path: Path | None = None
    wifi_rtt_ms: float | None = None
    error: str | None = None


DISPLAY_NAMES = {
    "efficientdet_lite0": "EfficientDet-Lite0",
    "nanodet-plus-m-1.5x_320": "NanoDet-Plus-M 1.5x",
    "picodet_s_320_coco": "PicoDet-S",
    "ssd_mobilenet_v2": "SSD MobileNet V2",
    "LeYOLONano": "LeYOLO Nano",
    "yolov5nu": "YOLOv5n-u",
    "yolov8n": "YOLOv8n",
    "yolov10n": "YOLOv10n",
    "yolo11n": "YOLO11n",
    "yolo26n": "YOLO26n",
}


SCRIPT_BY_MODEL = {
    "efficientdet_lite0": "EfficientDet_inference/main.py",
    "nanodet-plus-m-1.5x_320": "NanoDet-Plus_inference/main.py",
    "picodet_s_320_coco": "PicoDet-s_inference/main.py",
    "ssd_mobilenet_v2": "ssd_mobilenet_inference/v2_main.py",
    "LeYOLONano": "yolo_inference/main.py",
    "yolov5nu": "yolo_inference/main.py",
    "yolov8n": "yolo_inference/main.py",
    "yolov10n": "yolo_inference/main.py",
    "yolo11n": "yolo_inference/main.py",
    "yolo26n": "yolo_inference/main.py",
}

THRESHOLDS = {
    "efficientdet_lite0": 0.30,
    "nanodet-plus-m-1.5x_320": 0.45,
    "picodet_s_320_coco": 0.45,
    "ssd_mobilenet_v2": 0.55,
}

ROOT = Path(__file__).resolve().parent
SENDER_IP = "192.168.50.1"
PORT = 5000
CONTROL_PORT = 5001
RECEIVE_TIMEOUT = 10.0
SOURCE_FPS = 15.0
WARMUP = 10
THREADS = 4
OUTPUT_DIR = ROOT / "benchmark_results"
OUTPUT_CSV = OUTPUT_DIR / "results.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark ONNX models on Wi-Fi frames")
    parser.add_argument("--max-frames", type=int, default=400, help="Maximum frames to process (default: 400)")
    return parser.parse_args()


def element_count(shape: list[int | None] | None) -> int | None:
    if not shape or any(dim is None or dim <= 0 for dim in shape):
        return None
    return math.prod(shape)


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


def estimate_gmacs(model: Any) -> float | None:
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
            output_count = element_count(output_shape)
            if output_count is not None:
                total += output_count * math.prod(weight_shape[2:]) * weight_shape[1]
                found = True
        elif node.op_type in {"Gemm", "MatMul"}:
            lhs, rhs = shapes.get(node.input[0]), shapes.get(node.input[1])
            if lhs and rhs and len(lhs) >= 2 and len(rhs) >= 2 and lhs[-1] and rhs[-1]:
                total += lhs[-2] * lhs[-1] * rhs[-1]
                found = True
    return total / 1e9 if found else None


def load_module(path: Path, root: Path) -> ModuleType:
    name = f"benchmark_model_{path.stem}_{path.parent.name.replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(name, root / SCRIPT_BY_MODEL[path.stem])
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load inference script for {path.name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def benchmark_model(
    path: Path,
    root: Path,
    frames: list[Any],
    source_fps: float,
    wifi_rtt_ms: float | None,
    warmup: int,
    max_frames: int,
    threads: int,
    output_dir: Path | None,
) -> ModelRow:
    row = ModelRow(
        DISPLAY_NAMES.get(path.stem, path.stem), path,
        size_mb=path.stat().st_size / 1048576,
        wifi_rtt_ms=wifi_rtt_ms,
    )
    try:
        model = onnx.load(str(path), load_external_data=False)
        row.params_m = sum(math.prod(item.dims) for item in model.graph.initializer) / 1e6
        row.gmacs = estimate_gmacs(model)
        module = load_module(path, root)
        session = create_session(path.resolve(), threads)
        decode = getattr(module, "decode", None)
        if decode is None:
            decode = getattr(module, "decode_single", None) or getattr(module, "decode_pico", None)
        if decode is None:
            raise AttributeError(f"No decoder in {SCRIPT_BY_MODEL[path.stem]}")
        input_name = session.get_inputs()[0].name
        writer = None
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            safe_name = "".join(char if char.isalnum() or char in "-_" else "_" for char in row.name)
            row.video_path = output_dir / f"{safe_name}.mp4"
        timings: list[float] = []
        frame_index = 0
        try:
            while frame_index < len(frames) and (max_frames == 0 or row.frames < max_frames):
                frame = frames[frame_index].copy()
                start = time.perf_counter()
                input_tensor = module.preprocess(frame)
                outputs = session.run(None, {input_name: input_tensor})
                threshold = THRESHOLDS.get(path.stem, 0.35)
                detections = decode(outputs, frame.shape, threshold)
                elapsed = time.perf_counter() - start
                if output_dir is not None:
                    if writer is None:
                        height, width = frame.shape[:2]
                        writer = cv2.VideoWriter(
                            str(row.video_path), cv2.VideoWriter_fourcc(*"mp4v"),
                            source_fps, (width, height),
                        )
                        if not writer.isOpened():
                            raise RuntimeError(f"Cannot create video: {row.video_path}")
                    draw_detections(frame, detections)
                    current_fps = 1.0 / max(elapsed, 1e-9)
                    cv2.putText(
                        frame, f"{row.name} | conf {threshold:.2f} | FPS {current_fps:.2f} | latency {elapsed * 1000:.2f} ms",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 220), 2,
                    )
                    writer.write(frame)
                if frame_index >= warmup:
                    timings.append(elapsed)
                frame_index += 1
                row.frames += 1
        finally:
            if writer is not None:
                writer.release()
        if not timings:
            raise RuntimeError("Video has no measured frames after warmup")
        row.avg_latency_ms = sum(timings) / len(timings) * 1000
        row.min_latency_ms = min(timings) * 1000
        row.max_latency_ms = max(timings) * 1000
        fps = [1 / timing for timing in timings]
        row.avg_fps, row.min_fps, row.max_fps = sum(fps) / len(fps), min(fps), max(fps)
    except Exception as exc:
        row.error = f"{type(exc).__name__}: {exc}"
    return row


def fmt(value: float | None, digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def print_table(rows: list[ModelRow], source_name: str) -> None:
    headers = ["Model", "Frames", "FPS min", "FPS avg", "FPS max", "Latency avg ms", "Latency min ms", "Latency max ms", "Status"]
    values = []
    for row in rows:
        values.append([row.name, str(row.frames), fmt(row.min_fps), fmt(row.avg_fps), fmt(row.max_fps), fmt(row.avg_latency_ms), fmt(row.min_latency_ms), fmt(row.max_latency_ms), "OK" if not row.error else "ERR"])
    widths = [max(len(header), *(len(row[i]) for row in values)) for i, header in enumerate(headers)]
    border = "+" + "+".join("-" * (width + 2) for width in widths) + "+"
    print(f"Source: {source_name}")
    print(border)
    print("|" + "|".join(f" {header.ljust(width)} " for header, width in zip(headers, widths)) + "|")
    print(border)
    for row in values:
        print("|" + "|".join(f" {cell.ljust(width)} " for cell, width in zip(row, widths)) + "|")
    print(border)
    for row in rows:
        if row.error:
            print(f"{row.name}: {row.error}")
        elif row.video_path:
            print(f"{row.name} video: {row.video_path}")


def write_csv(rows: list[ModelRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["model", "path", "frames", "fps_min", "fps_avg", "fps_max", "latency_avg_ms", "latency_min_ms", "latency_max_ms", "wifi_rtt_ms", "params_m", "gmacs", "error"]
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "model": row.name, "path": str(row.path), "frames": row.frames, "fps_min": row.min_fps, "fps_avg": row.avg_fps, "fps_max": row.max_fps,
                "latency_avg_ms": row.avg_latency_ms, "latency_min_ms": row.min_latency_ms, "latency_max_ms": row.max_latency_ms, "wifi_rtt_ms": row.wifi_rtt_ms, "params_m": row.params_m, "gmacs": row.gmacs, "error": row.error or "",
            })


def main() -> None:
    args = parse_args()
    if args.max_frames <= 0:
        raise SystemExit("--max-frames must be greater than zero")
    frames, wifi_rtt_ms = load_wifi_frames(
        SENDER_IP, PORT, args.max_frames, RECEIVE_TIMEOUT, CONTROL_PORT,
    )
    source_fps = SOURCE_FPS
    source_name = f"wifi://{SENDER_IP}:{PORT}"
    if not frames:
        raise SystemExit("No frames received from source")
    paths = [path for path in sorted(ROOT.rglob("*.onnx")) if "venv" not in path.parts and path.stem in SCRIPT_BY_MODEL]
    if not paths:
        raise SystemExit(f"No supported ONNX models found under {ROOT}")
    print(f"Benchmark: {source_name}; frames={len(frames)}; warmup={WARMUP}; threads={THREADS}")
    rows = []
    for index, path in enumerate(paths, 1):
        print(f"[{index}/{len(paths)}] {DISPLAY_NAMES.get(path.stem, path.stem)}...", flush=True)
        rows.append(benchmark_model(path, ROOT, frames, source_fps, wifi_rtt_ms, WARMUP, args.max_frames, THREADS, OUTPUT_DIR))
    print_table(rows, source_name)
    write_csv(rows, OUTPUT_CSV)
    print(f"CSV written to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
