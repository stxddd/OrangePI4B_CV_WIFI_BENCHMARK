import argparse
import os
import socket
import struct
import threading
import time
from pathlib import Path
from typing import Tuple

import cv2


MAGIC = b"OPV1"
HEADER = struct.Struct("!4sIHH")
DEFAULT_PORT = 5000
CONTROL_PORT = 5001
DEFAULT_CHUNK_SIZE = 1200
CAMERA_DEVICE = "/dev/video0"
RECEIVER_IP = "192.168.50.2"
FRAME_WIDTH = 320
FRAME_HEIGHT = 240
FRAME_RATE = 15.0
JPEG_QUALITY = 75
PING = b"OP-PING"
PONG = b"OP-PONG"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send camera or video frames over UDP")
    parser.add_argument("source", choices=("camera", "video"))
    parser.add_argument("video", type=Path, nargs="?", help="Video file used with video mode")
    return parser.parse_args()


def open_capture(args: argparse.Namespace) -> cv2.VideoCapture:
    if args.source == "video":
        if args.video is None:
            raise SystemExit("Video path is required in video mode")
        if not args.video.exists():
            raise SystemExit(f"Video not found: {args.video}")
        capture = cv2.VideoCapture(str(args.video))
    else:
        camera_source: int | str = CAMERA_DEVICE
        if os.name == "nt":
            capture = cv2.VideoCapture(camera_source)
        else:
            capture = cv2.VideoCapture(camera_source, cv2.CAP_V4L2)
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        capture.set(cv2.CAP_PROP_FPS, FRAME_RATE)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {args.source}: {args.video if args.source == 'video' else CAMERA_DEVICE}")
    return capture


def send_frame(sock: socket.socket, address: Tuple[str, int], frame_id: int, jpeg: bytes, chunk_size: int) -> None:
    chunks = [jpeg[offset:offset + chunk_size] for offset in range(0, len(jpeg), chunk_size)]

    for chunk_id, chunk in enumerate(chunks):
        packet = HEADER.pack(MAGIC, frame_id, chunk_id, len(chunks)) + chunk
        sock.sendto(packet, address)


def latency_responder(control_port: int) -> None:
    control_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    control_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    control_socket.bind(("0.0.0.0", control_port))
    while True:
        packet, address = control_socket.recvfrom(64)
        if packet == PING:
            control_socket.sendto(PONG, address)


def main() -> None:
    args = parse_args()
    capture = open_capture(args)

    address = (RECEIVER_IP, DEFAULT_PORT)
    source_name = CAMERA_DEVICE if args.source == "camera" else args.video
    print(f"Источник {source_name} открыт; отправка на {address[0]}:{address[1]}", flush=True)
    threading.Thread(target=latency_responder, args=(CONTROL_PORT,), daemon=True).start()
    interval = 1.0 / FRAME_RATE
    frame_id = 0
    encode_options = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    try:
        next_frame_at = time.monotonic()
        while True:
            ok, frame = capture.read()
            if not ok:
                if args.source == "video":
                    print("Видео закончилось", flush=True)
                    break
                raise RuntimeError("camera.read() error")
            encoded, jpeg = cv2.imencode(".jpg", frame, encode_options)
            if not encoded:
                continue
            send_frame(sock, address, frame_id, jpeg.tobytes(), DEFAULT_CHUNK_SIZE)
            frame_id = (frame_id + 1) & 0xFFFFFFFF
            next_frame_at += interval
            delay = next_frame_at - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_frame_at = time.monotonic()
    finally:
        capture.release()
        sock.close()


if __name__ == "__main__":
    main()
