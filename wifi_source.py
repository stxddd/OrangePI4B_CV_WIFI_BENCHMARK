from __future__ import annotations

import socket
import struct
import time

import cv2
import numpy as np


MAGIC = b"OPV1"
HEADER = struct.Struct("!4sIHH")
CONTROL_PORT = 5001
PING = b"OP-PING"
PONG = b"OP-PONG"
MAX_FRAME_BYTES = 4 * 1024 * 1024
RECEIVE_BUFFER = 65535


def receive_jpeg(
    sock: socket.socket,
    sender_ip: str,
    parts: dict[int, dict[int, bytes]],
    sizes: dict[int, int],
) -> bytes:
    while True:
        packet, address = sock.recvfrom(RECEIVE_BUFFER)
        if address[0] != sender_ip or len(packet) < HEADER.size:
            continue
        magic, frame_id, chunk_id, chunk_count = HEADER.unpack_from(packet)
        if magic != MAGIC or chunk_count == 0 or chunk_id >= chunk_count:
            continue
        frame_parts = parts.setdefault(frame_id, {})
        if chunk_id not in frame_parts:
            frame_parts[chunk_id] = packet[HEADER.size:]
            sizes[frame_id] = sizes.get(frame_id, 0) + len(frame_parts[chunk_id])
        if sizes[frame_id] > MAX_FRAME_BYTES:
            parts.pop(frame_id, None)
            sizes.pop(frame_id, None)
            continue
        if len(frame_parts) == chunk_count:
            jpeg = b"".join(frame_parts[index] for index in range(chunk_count))
            parts.clear()
            sizes.clear()
            return jpeg
        if len(parts) > 30:
            oldest_frame = next(iter(parts))
            parts.pop(oldest_frame, None)
            sizes.pop(oldest_frame, None)


def measure_rtt(sender_ip: str, control_port: int = CONTROL_PORT) -> float | None:
    probe_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    probe_socket.settimeout(0.5)
    started = time.monotonic()
    try:
        probe_socket.sendto(PING, (sender_ip, control_port))
        response, _ = probe_socket.recvfrom(32)
        if response != PONG:
            return None
        return (time.monotonic() - started) * 1000.0
    except socket.timeout:
        return None
    finally:
        probe_socket.close()


def load_wifi_frames(
    sender_ip: str,
    port: int,
    frame_count: int,
    receive_timeout: float,
    control_port: int = CONTROL_PORT,
) -> tuple[list[np.ndarray], float | None]:
    if frame_count <= 0:
        raise ValueError("--max-frames must be greater than zero")

    frames: list[np.ndarray] = []
    parts: dict[int, dict[int, bytes]] = {}
    sizes: dict[int, int] = {}
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, MAX_FRAME_BYTES)
        sock.settimeout(receive_timeout)
        sock.bind(("0.0.0.0", port))
        print(f"Ожидание Wi-Fi-кадров от {sender_ip}:{port}", flush=True)
        while len(frames) < frame_count:
            try:
                jpeg = receive_jpeg(sock, sender_ip, parts, sizes)
            except socket.timeout as exc:
                raise TimeoutError("Не дождались полного JPEG-кадра по Wi-Fi") from exc
            frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                frames.append(frame)
    return frames, measure_rtt(sender_ip, control_port)

