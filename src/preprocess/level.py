"""Horizon tilt detection via OpenCV Hough-lines.

Samples a handful of frames from a clip, runs Canny + HoughLinesP on
each, keeps lines that are within ±20° of horizontal, and returns the
median tilt angle (degrees, positive = counter-clockwise rotation
needed to level).
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_SAMPLE_FRAMES = 8
_NEAR_HORIZONTAL_DEG = 20.0
_MIN_LINE_FRAC = 0.25  # line must span ≥ 25 % of frame width


def detect_horizon_tilt(path: Path) -> float:
    """Return the median horizon tilt of *path* in degrees (0 if unsure).

    Positive values indicate the frame is rotated clockwise (right side
    down) — so ``rotate=-tilt`` levels it.

    Args:
        path: Video file to analyse.

    Returns:
        Tilt angle in degrees.  ``0.0`` if no dominant horizontal line
        was found or the clip can't be opened.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return 0.0
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        if n <= 0 or w <= 0:
            return 0.0

        step = max(1, n // _SAMPLE_FRAMES)
        min_len = int(w * _MIN_LINE_FRAC)
        angles: list[float] = []

        for i in range(0, n, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, frame = cap.read()
            if not ok:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 60, 180)
            lines = cv2.HoughLinesP(
                edges, rho=1, theta=math.pi / 360,
                threshold=80, minLineLength=min_len, maxLineGap=20,
            )
            if lines is None:
                continue
            for x1, y1, x2, y2 in lines[:, 0]:
                if x2 == x1:
                    continue
                deg = math.degrees(math.atan2(y2 - y1, x2 - x1))
                # Collapse to the −90°…+90° half-plane.
                if deg > 90:
                    deg -= 180
                elif deg < -90:
                    deg += 180
                if abs(deg) <= _NEAR_HORIZONTAL_DEG:
                    angles.append(deg)
    finally:
        cap.release()

    if not angles:
        return 0.0
    return float(np.median(angles))
