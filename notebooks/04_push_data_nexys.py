#!/usr/bin/env python3
"""Send Fashion-MNIST images to the Nexys over UART and compare with float32.

Protocol (115200 baud, 8N1, one image at a time):
  PC -> FPGA : 784 signed 16-bit little-endian words (1568 bytes)
  FPGA -> PC : 1 class byte, then 10 signed 16-bit little-endian scores (21 bytes)

Example:
  python notebooks/04_push_data_nexys.py --port COM5 --images 100
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import time

import numpy as np

CLASSES = (
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
)
INPUT_WORDS = 784
OUTPUT_WORDS = 10
RESPONSE_BYTES = 1 + 2 * OUTPUT_WORDS
EXPORT_DIR = Path(__file__).resolve().parents[1] / "data" / "quantized_data" / "fashion" / "nexys" / "sequential"


def pack_image(x_int: np.ndarray) -> bytes:
    if x_int.shape != (INPUT_WORDS,):
        raise ValueError(f"Expected {INPUT_WORDS} input words, got shape {x_int.shape}")
    return np.asarray(x_int, dtype="<i2").tobytes()


def unpack_response(payload: bytes) -> tuple[int, np.ndarray]:
    if len(payload) != RESPONSE_BYTES:
        raise ValueError(f"Expected {RESPONSE_BYTES} bytes, got {len(payload)}")
    return payload[0], np.frombuffer(payload[1:], dtype="<i2").astype(np.int16)


def read_exact(port, n_bytes: int, timeout_s: float) -> bytes:
    deadline = time.perf_counter() + timeout_s
    payload = bytearray()
    while len(payload) < n_bytes and time.perf_counter() < deadline:
        payload += port.read(n_bytes - len(payload))
    if len(payload) != n_bytes:
        raise TimeoutError(
            f"UART timeout: received {len(payload)} of {n_bytes} bytes. "
            "Check the port, the bitstream and press the reset button."
        )
    return bytes(payload)


def infer(port, x_int: np.ndarray, timeout_s: float) -> tuple[int, np.ndarray, float]:
    """Run one complete exchange and return the class, scores and duration."""
    start = time.perf_counter()
    port.write(pack_image(x_int))
    port.flush()
    payload = read_exact(port, RESPONSE_BYTES, timeout_s)
    duration = time.perf_counter() - start
    return (*unpack_response(payload), duration)


def run(port, dataset: Path, images: int, timeout_s: float, output_csv: Path) -> bool:
    with np.load(dataset) as data:
        x_int = data["x_int"][:images]
        labels = data["y"][:images]
        class_float = data["class_float"][:images]
        expected_scores = data["pred_board_int"][:images]

    rows = []
    for index in range(len(x_int)):
        board_class, scores, duration = infer(port, x_int[index], timeout_s)
        rows.append(
            {
                "image": index,
                "label": int(labels[index]),
                "class_float": int(class_float[index]),
                "class_fpga": board_class,
                "class_matches_scores": board_class == int(np.argmax(scores)),
                "scores_match_simulation": bool(np.array_equal(scores, expected_scores[index])),
                "duration_ms": duration * 1000,
            }
        )
        row = rows[-1]
        print(
            f"{index:5d} | vraie : {CLASSES[row['label']]:11s} | float32 : {CLASSES[row['class_float']]:11s} "
            f"| FPGA : {CLASSES[board_class]:11s} | {row['duration_ms']:7.1f} ms"
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    count = len(rows)
    same_class = sum(row["class_fpga"] == row["class_float"] for row in rows)
    correct = sum(row["class_fpga"] == row["label"] for row in rows)
    coherent = sum(row["class_matches_scores"] for row in rows)
    same_scores = sum(row["scores_match_simulation"] for row in rows)
    mean_ms = sum(row["duration_ms"] for row in rows) / count

    print(f"\nImages                        : {count}")
    print(f"FPGA identique au float32     : {same_class}/{count}")
    print(f"Précision du FPGA             : {correct / count:.2%}")
    print(f"Classe cohérente avec scores  : {coherent}/{count}")
    print(f"Scores identiques simulation  : {same_scores}/{count}")
    print(f"Durée moyenne d'un échange    : {mean_ms:.1f} ms ({1000 / mean_ms:.1f} images/s)")
    print(f"Résultats                     : {output_csv}")
    return same_scores == count and coherent == count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", help="Serial port, for example COM5")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--images", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=5.0, help="Timeout per image in seconds")
    parser.add_argument("--dataset", type=Path, default=EXPORT_DIR / "dataset_fashion_quantized.npz")
    parser.add_argument("--output", type=Path, default=EXPORT_DIR / "fpga_results.csv")
    parser.add_argument("--list-ports", action="store_true")
    args = parser.parse_args()

    import serial
    import serial.tools.list_ports

    if args.list_ports or not args.port:
        for found in serial.tools.list_ports.comports():
            print(f"{found.device}: {found.description}")
        return 0 if args.list_ports else 2

    with serial.Serial(args.port, args.baud, timeout=0.05, write_timeout=args.timeout) as port:
        time.sleep(0.05)
        port.reset_input_buffer()
        return 0 if run(port, args.dataset, args.images, args.timeout, args.output) else 1


if __name__ == "__main__":
    raise SystemExit(main())
