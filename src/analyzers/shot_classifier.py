"""Shot-type + camera-movement classifier.

Combines two signals:

* **Shot type** — YOLOv8 (``ultralytics``) detects the dominant subject
  per sampled frame.  The ratio of subject bounding-box area to frame
  area maps to ``wide_establishing`` / ``mid`` / ``detail``.  When
  ``ultralytics`` is not installed the shot type falls back to
  ``None`` and only movement classification runs.

* **Camera movement** — dense optical flow (Farnebäck) sampled across
  the clip.  Aggregate magnitude and direction statistics map to
  ``static``, ``orbit``, ``push_in``, ``pull_out``, ``reveal`` (tilt
  up), ``top_down``, ``tracking``, or ``fly_through``.

Both tags are stored on ``Clip.metadata["shot_type"]`` and
``Clip.metadata["movement"]`` — the beat-aware sequencer already reads
those fields and will start biasing picks.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from src.analyzers.base import BaseAnalyzer
from src.models.clip import Clip

logger = logging.getLogger(__name__)

_SAMPLE_FPS = 2.0          # frames per second sampled during analysis
_FLOW_DOWNSCALE = (320, 180)  # resize flow inputs for speed
_MIN_SAMPLES = 4           # below this we give up classifying

# Shot-type thresholds: subject bbox / frame area.
_SHOT_WIDE_MAX = 0.20      # <20 % = wide_establishing
_SHOT_DETAIL_MIN = 0.60    # >60 % = detail

# Movement classifier magnitude thresholds (px / frame, downscaled).
_STATIC_MAG = 0.4
_HIGH_MAG = 3.5

# YOLO is loaded lazily — only on the first call that needs it.
_YOLO_MODEL: Any = None
_YOLO_READY: bool | None = None  # tri-state: None=unknown, True=ready, False=failed


class ShotClassifierAnalyzer(BaseAnalyzer):
    """Assign ``shot_type`` and ``movement`` tags to each clip."""

    @property
    def name(self) -> str:
        """Analyzer name used in ``enabled_analyzers``."""
        return "shot_classifier"

    def analyze(self, clip: Clip) -> Clip:
        """Sample *clip*, classify shot type + movement, attach metadata."""
        cap = cv2.VideoCapture(str(clip.path))
        if not cap.isOpened():
            logger.warning("shot_classifier: could not open %s", clip.path)
            return clip

        try:
            frames, times = _sample_frames(cap, clip.duration, _SAMPLE_FPS)
        finally:
            cap.release()

        if len(frames) < _MIN_SAMPLES:
            return clip

        shot_type = _classify_shot_type(frames)
        movement, flow_stats = _classify_movement(frames)

        if shot_type:
            clip = clip.with_metadata("shot_type", shot_type)
            clip = clip.with_tag(shot_type)
        if movement:
            clip = clip.with_metadata("movement", movement)
            clip = clip.with_tag(movement)
        if flow_stats:
            clip = clip.with_metadata("flow_stats", flow_stats)

        logger.info(
            "%s — shot=%s  movement=%s  flow_mag=%.2f",
            clip.path.name, shot_type or "—", movement or "—",
            flow_stats.get("mean_mag", 0.0),
        )
        return clip


# ------------------------------------------------------------------
# Frame sampling
# ------------------------------------------------------------------

def _sample_frames(
    cap: cv2.VideoCapture,
    duration: float,
    fps: float,
) -> tuple[list[np.ndarray], list[float]]:
    """Pull ~*fps* frames per second from an open VideoCapture."""
    n = max(2, int(round(duration * fps)))
    times = np.linspace(0.0, max(0.0, duration - 0.05), n).tolist()

    frames: list[np.ndarray] = []
    kept_times: list[float] = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000.0)
        ok, frame = cap.read()
        if ok and frame is not None:
            frames.append(frame)
            kept_times.append(float(t))
    return frames, kept_times


# ------------------------------------------------------------------
# Shot type (YOLOv8)
# ------------------------------------------------------------------

def _classify_shot_type(frames: list[np.ndarray]) -> str | None:
    """Return the dominant shot type across *frames*, or None if YOLO unavailable."""
    model = _load_yolo()
    if model is None:
        return None

    ratios: list[float] = []
    try:
        # Run YOLO on each frame; low verbosity.
        results = model.predict(frames, verbose=False, conf=0.25)
    except Exception:
        logger.exception("YOLO inference failed — skipping shot-type classification")
        return None

    for res in results:
        try:
            boxes = getattr(res, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue
            xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.asarray(boxes.xyxy)
            if xyxy.size == 0:
                continue
            frame_area = float(res.orig_shape[0] * res.orig_shape[1])
            if frame_area <= 0:
                continue
            # Largest box area as the dominant subject.
            areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
            ratios.append(float(areas.max()) / frame_area)
        except Exception:
            continue

    if not ratios:
        return None

    mean_ratio = float(np.mean(ratios))
    if mean_ratio < _SHOT_WIDE_MAX:
        return "wide_establishing"
    if mean_ratio > _SHOT_DETAIL_MIN:
        return "detail"
    return "mid"


def _load_yolo() -> Any:
    """Lazy-load a small YOLOv8 model, caching failure to avoid repeated attempts."""
    global _YOLO_MODEL, _YOLO_READY
    if _YOLO_READY is False:
        return None
    if _YOLO_MODEL is not None:
        return _YOLO_MODEL
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception:
        logger.info("ultralytics not installed — shot-type classification disabled")
        _YOLO_READY = False
        return None
    try:
        _YOLO_MODEL = YOLO("yolov8n.pt")
        _YOLO_READY = True
        logger.info("YOLOv8n model loaded for shot classification")
    except Exception:
        logger.exception("YOLOv8 model failed to load — disabling")
        _YOLO_READY = False
        return None
    return _YOLO_MODEL


# ------------------------------------------------------------------
# Camera movement (dense optical flow)
# ------------------------------------------------------------------

def _classify_movement(frames: list[np.ndarray]) -> tuple[str, dict[str, float]]:
    """Return a movement tag + per-clip aggregate flow statistics."""
    grays = [cv2.cvtColor(cv2.resize(f, _FLOW_DOWNSCALE), cv2.COLOR_BGR2GRAY)
             for f in frames]

    mean_dx_vals: list[float] = []
    mean_dy_vals: list[float] = []
    mean_mag_vals: list[float] = []
    radial_vals: list[float] = []

    h, w = grays[0].shape
    cy, cx = h / 2.0, w / 2.0
    # Pre-compute pixel coordinates relative to centre for radial dot-product.
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    ry = ys - cy
    rx = xs - cx
    norm = np.sqrt(rx ** 2 + ry ** 2) + 1e-6

    for i in range(1, len(grays)):
        flow = cv2.calcOpticalFlowFarneback(
            grays[i - 1], grays[i],
            None, 0.5, 3, 15, 3, 5, 1.2, 0,
        )
        dx = flow[..., 0]
        dy = flow[..., 1]
        mag = np.sqrt(dx ** 2 + dy ** 2)

        mean_dx_vals.append(float(dx.mean()))
        mean_dy_vals.append(float(dy.mean()))
        mean_mag_vals.append(float(mag.mean()))

        # Radial component: positive = push_in (vectors point outward from centre),
        # negative = pull_out.
        dot = (dx * rx + dy * ry) / norm
        radial_vals.append(float(dot.mean()))

    if not mean_mag_vals:
        return "static", {}

    mean_mag = float(np.mean(mean_mag_vals))
    mean_dx = float(np.mean(mean_dx_vals))
    mean_dy = float(np.mean(mean_dy_vals))
    mean_radial = float(np.mean(radial_vals))
    stats = {
        "mean_mag": round(mean_mag, 3),
        "mean_dx": round(mean_dx, 3),
        "mean_dy": round(mean_dy, 3),
        "mean_radial": round(mean_radial, 3),
    }

    # Classify.
    if mean_mag < _STATIC_MAG:
        return "static", stats

    # Strong radial motion → zoom.
    if abs(mean_radial) > 0.5 and abs(mean_radial) > abs(mean_dx) and abs(mean_radial) > abs(mean_dy):
        return ("push_in" if mean_radial > 0 else "pull_out"), stats

    # Dominant vertical motion.
    if abs(mean_dy) > abs(mean_dx) * 1.3 and abs(mean_dy) > 0.4:
        if mean_dy < 0:
            return "reveal", stats   # camera tilting up = scene moves down
        return "top_down", stats

    # Dominant horizontal motion.
    if abs(mean_dx) > abs(mean_dy) * 1.3 and abs(mean_dx) > 0.4:
        return ("tracking" if mean_mag > _HIGH_MAG else "orbit"), stats

    # High overall magnitude with no clear axis → fly_through.
    if mean_mag >= _HIGH_MAG:
        return "fly_through", stats

    return "static", stats
