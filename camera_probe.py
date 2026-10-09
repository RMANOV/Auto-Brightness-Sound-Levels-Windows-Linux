#!/usr/bin/env python3
"""Bounded ambient measurement; persisted reference means a verified adjustment.

Camera frames stay in memory. The caller must also enforce a process timeout,
because device open/read/release can block below Python's monotonic deadline.
"""
import argparse
import json
import math
import os
from pathlib import Path
import statistics
import sys
import tempfile
import time

SOURCE = "v4l2-gray-warmup3-median3-v1"


def validate_sample(sample):
    if not isinstance(sample, dict) or set(sample) != {"schema", "source", "ambient", "captured_at"}:
        raise ValueError("Invalid ambient record")
    if type(sample["schema"]) is not int or sample["schema"] != 1 or sample["source"] != SOURCE:
        raise ValueError("Incompatible ambient measurement")
    for key in ("ambient", "captured_at"):
        value = sample[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("Non-finite ambient record")
    if not 0 <= sample["ambient"] <= 100 or sample["captured_at"] < 0:
        raise ValueError("Ambient record out of range")
    return sample


def measure(cv, clock=time.monotonic, wall_clock=time.time):
    start = clock()
    cap = cv.VideoCapture(0, cv.CAP_V4L2)
    try:
        if not cap.isOpened():
            raise RuntimeError("Camera unavailable")
        if not (cap.set(cv.CAP_PROP_FRAME_WIDTH, 320) and cap.set(cv.CAP_PROP_FRAME_HEIGHT, 240)):
            raise RuntimeError("Camera measurement format unavailable")
        values = []
        for index in range(6):
            if clock() - start > 2.5:
                raise TimeoutError("Camera sample deadline exceeded")
            ok, frame = cap.read()
            if clock() - start > 2.5:
                raise TimeoutError("Camera sample deadline exceeded")
            if not ok or frame is None or frame.size == 0:
                raise RuntimeError("Camera returned no frame")
            value = float(cv.cvtColor(frame, cv.COLOR_BGR2GRAY).mean()) / 255 * 100
            if not math.isfinite(value) or not 0 <= value <= 100:
                raise ValueError("Camera returned an invalid measurement")
            if index >= 3:
                values.append(value)
        result = validate_sample({"schema": 1, "source": SOURCE,
                                  "ambient": statistics.median(values),
                                  "captured_at": wall_clock()})
        if clock() - start > 2.5:
            raise TimeoutError("Camera sample processing deadline exceeded")
        return result
    finally:
        cap.release()


def changed(sample, baseline):
    validate_sample(sample)
    try:
        validate_sample(baseline)
        age = sample["captured_at"] - baseline["captured_at"]
        if not 0 <= age <= 86400:
            return True
    except (TypeError, ValueError):
        return True
    # Keep the last verified adjustment as baseline so gradual changes accumulate.
    return abs(sample["ambient"] - baseline["ambient"]) > .4 * max(baseline["ambient"], 1)


def read_baseline(path):
    try:
        return read_json(path)
    except (OSError, ValueError):
        return None


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate measurement field")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=unique)


def accept(path, sample, now=None):
    validate_sample(sample)
    age = (time.time() if now is None else now) - sample["captured_at"]
    if not 0 <= age <= 600:
        raise ValueError("Cannot acknowledge stale/future measurement")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".ambient-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(sample, stream, allow_nan=False)
            stream.write("\n")
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--accept", type=Path)
    parser.add_argument("--validate", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.accept or args.validate:
            result = read_json(args.accept or args.validate)
            if not isinstance(result, dict) or set(result) != {"status", "sample"} or result["status"] not in {"CHANGED", "UNCHANGED"}:
                raise ValueError("Cannot acknowledge an unsuccessful measurement")
            validate_sample(result["sample"])
            if args.validate:
                if not 0 <= time.time() - result["sample"]["captured_at"] <= 30:
                    raise ValueError("Measurement is not fresh")
                print(result["status"])
                return 0
            if result["status"] != "CHANGED":
                raise ValueError("No verified adjustment to acknowledge")
            accept(args.state, result["sample"])
            return 0
        import cv2  # Hardware library is loaded only in the explicit probe path.
        sample = measure(cv2)
        status = "CHANGED" if changed(sample, read_baseline(args.state)) else "UNCHANGED"
        print(json.dumps({"status": status, "sample": sample}, allow_nan=False), flush=True)
        return 0 if status == "CHANGED" else 1
    except Exception as exc:
        print(json.dumps({"status": "ERROR", "error": str(exc)}), flush=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
