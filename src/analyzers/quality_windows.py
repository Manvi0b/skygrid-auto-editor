"""Per-window video quality analyzer.

Samples each clip at ~1-second intervals and scores four independent
metrics on every sample:

* **Sharpness** — Laplacian variance (focus / softness detector).
* **Exposure** — histogram clipping at the top and bottom of the range.
* **Stability** — frame-to-frame pixel difference (jitter detector).
* **Horizon** — absolute tilt angle of the dominant horizontal edge.

Each metric is normalised to 0–100 and combined into a per-window
quality score.  Results are attached to ``Clip.metadata["windows"]``
as a list of ``{"t": float, "score": float, ...}`` dicts so downstream
stages (segment extraction) can walk them.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from src.analyzers.base import BaseAnalyzer
from src.models.clip import Clip

logger = logging.getLogger(__name__)

_WINDOW_SECONDS = 1.0
_FRAMES_PER_WINDOW = 3         # sample 3 frames per 1-second window
_MIN_CLIP_DURATION = 1.5       # clips shorter than this are skipped entirely
_SHARPNESS_SAT = 500.0         # Laplacian variance that maps to 100/100
_STABILITY_FLOOR = 20.0        # mean abs diff below this is "perfectly still"


class QualityWindowsAnalyzer(BaseAnalyzer):
    """Score video quality in 1-second windows, not just per clip."""

    @property
    def name(self) -> str:
        """Analyzer identifier used in config ``enabled_analyzers``."""
        return "quality_windows"

    def analyze(self, clip: Clip) -> Clip:
        """Walk *clip* frame-by-frame, emit a list of per-window scores.

        Attaches ``metadata["windows"]`` — a list of dicts with keys
        ``t``, ``score``, ``sharpness``, ``exposure``, ``stability``,
        ``horizon``.  Also sets ``scores["quality_windows"]`` to the
        mean window score for compatibility with the existing filter
        path.
        """
        if clip.duration < _MIN_CLIP_DURATION:
            return clip

        cap = cv2.VideoCapture(str(clip.path))
        if not cap.isOpened():
            logger.warning("quality_windows: could not open %s", clip.path)
            return clip

        try:
            windows = _score_clip(cap, clip.duration)
        finally:
            cap.release()

        if not windows:
            return clip

        mean_score = float(np.mean([w["score"] for w in windows]))
        clip = clip.with_metadata("windows", windows)
        clip = clip.with_score(self.name, mean_score)

        # Tag low-quality clips so existing filter paths still catch them.
        if mean_score < 40.0:
            clip = clip.with_tag("low_quality")
        return clip


# ------------------------------------------------------------------
# Core scoring loop
# ------------------------------------------------------------------

def _score_clip(cap: "cv2.VideoCapture", duration: float) -> list[dict[str, Any]]:
    """Return a list of per-window score dicts for an open VideoCapture."""
    n_windows = max(1, int(round(duration / _WINDOW_SECONDS)))
    windows: list[dict[str, Any]] = []

    prev_gray_small: np.ndarray | None = None

    for i in range(n_windows):
        t0 = i * _WINDOW_SECONDS
        t1 = min(duration, (i + 1) * _WINDOW_SECONDS)
        mid_times = np.linspace(t0, t1, _FRAMES_PER_WINDOW, endpoint=False) + (
            (t1 - t0) / (2 * _FRAMES_PER_WINDOW)
        )

        sharp_vals: list[float] = []
        expo_vals: list[float] = []
        stab_vals: list[float] = []
        horiz_vals: list[float] = []

        for t in mid_times:
            cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (320, 180))

            sharp_vals.append(_sharpness(gray))
            expo_vals.append(_exposure(gray))
            horiz_vals.append(_horizon_tilt(small))

            if prev_gray_small is not None and prev_gray_small.shape == small.shape:
                stab_vals.append(_stability(prev_gray_small, small))
            prev_gray_small = small

        # Aggregate (mean of sub-samples) with sensible fallbacks.
        sharpness = float(np.mean(sharp_vals)) if sharp_vals else 0.0
        exposure = float(np.mean(expo_vals)) if expo_vals else 0.0
        stability = float(np.mean(stab_vals)) if stab_vals else 100.0
        horizon = float(np.mean(horiz_vals)) if horiz_vals else 100.0

        # Weighted combination — sharpness & stability dominate.
        score = (
            0.35 * sharpness
            + 0.25 * exposure
            + 0.25 * stability
            + 0.15 * horizon
        )

        windows.append({
            "t": round(float(t0), 3),
            "end": round(float(t1), 3),
            "score": round(score, 2),
            "sharpness": round(sharpness, 2),
            "exposure": round(exposure, 2),
            "stability": round(stability, 2),
            "horizon": round(horizon, 2),
        })

    return windows


# ------------------------------------------------------------------
# Metric primitives
# ------------------------------------------------------------------

def _sharpness(gray: np.ndarray) -> float:
    """Laplacian variance → 0–100 (saturates at ``_SHARPNESS_SAT``)."""
    var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return float(min(100.0, (var / _SHARPNESS_SAT) * 100.0))


def _exposure(gray: np.ndarray) -> float:
    """Penalise blown highlights + crushed shadows. Returns 0–100."""
    total = gray.size
    if total == 0:
        return 0.0
    blown = float(np.count_nonzero(gray >= 250)) / total
    crushed = float(np.count_nonzero(gray <= 5)) / total
    # >5 % blown OR crushed → sharp score drop.
    penalty = min(1.0, (blown * 4.0) + (crushed * 3.0))
    return float((1.0 - penalty) * 100.0)


def _stability(prev: np.ndarray, curr: np.ndarray) -> float:
    """Inverse mean-abs-diff between consecutive frames → 0–100."""
    diff = cv2.absdiff(prev, curr)
    mad = float(diff.mean())
    # Linear taper: 0 MAD = 100, 60+ MAD = 0.
    scaled = max(0.0, 1.0 - (mad - _STABILITY_FLOOR) / 40.0)
    return float(min(1.0, scaled) * 100.0)


def _horizon_tilt(gray_small: np.ndarray) -> float:
    """Detect horizon tilt via Hough lines — 0° tilt = 100, >5° = 0."""
    try:
        edges = cv2.Canny(gray_small, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=120)
        if lines is None:
            return 100.0
        # Find the most horizontal-ish line (theta near pi/2).
        best = 90.0
        for rho_theta in lines[:20]:
            _, theta = rho_theta[0]
            deg = abs(np.degrees(theta) - 90.0)
            if deg < best:
                best = float(deg)
        # 0° deviation → 100, 5°+ → 0.
        return float(max(0.0, 1.0 - best / 5.0) * 100.0)
    except Exception:
        return 100.0
