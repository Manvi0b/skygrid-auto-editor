"""Smart reframe — subject-tracked cropping for horizontal→vertical conversion.

A centre-crop when going from 16:9 to 9:16 loses the edges of the
frame and often cuts the subject in half.  This module samples the
source, locates the dominant subject at each sample, smooths the path
with a low-pass filter, and crops around the subject path per frame.

Detection strategy (first match wins):

1. **YOLOv8** (``ultralytics``) — the largest detected subject bbox.
2. **Saliency fallback** — row/column projection of grayscale gradient
   magnitude; the centroid of the top 20 % is used.
3. **Rule-of-thirds fallback** — if nothing else produces a signal,
   anchor the crop window on the rule-of-thirds intersection closest
   to the frame centre.

The result is a MoviePy clip at the exact target dimensions, with
cropping varying over time.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import cv2
import numpy as np
from moviepy import VideoFileClip
from moviepy.video.fx import Resize

logger = logging.getLogger(__name__)

_SAMPLE_PERIOD = 0.5         # seconds between subject samples
_SMOOTH_WINDOW = 5           # moving-average window over samples (odd is best)
_THIRD = 1.0 / 3.0

_YOLO_MODEL: Any = None
_YOLO_READY: bool | None = None


def smart_reframe(
    vc: VideoFileClip,
    tgt_w: int,
    tgt_h: int,
) -> VideoFileClip:
    """Return a clip cropped to (tgt_w, tgt_h) tracking the dominant subject.

    Args:
        vc: Source MoviePy clip.
        tgt_w: Target width in pixels.
        tgt_h: Target height in pixels.

    Returns:
        A clip rendered at exactly (tgt_w, tgt_h) with time-varying
        crop offsets.
    """
    src_w, src_h = vc.w, vc.h
    tgt_ar = tgt_w / max(tgt_h, 1)

    # Determine the crop rectangle size in source coordinates — we take the
    # largest rect with the target aspect that fits inside the source.
    if src_w / src_h > tgt_ar:
        # Source is wider: crop horizontally, full height.
        crop_h = src_h
        crop_w = int(src_h * tgt_ar)
    else:
        # Source is taller: crop vertically, full width.
        crop_w = src_w
        crop_h = int(src_w / tgt_ar)

    path = _compute_subject_path(vc, crop_w, crop_h)
    centre_fn = _smoothed_centre_fn(path, vc.duration, src_w, src_h)

    def _frame_transform(get_frame: Callable[[float], np.ndarray], t: float) -> np.ndarray:
        frame = get_frame(t)
        h, w = frame.shape[:2]
        cx, cy = centre_fn(t)
        x1 = int(np.clip(cx - crop_w / 2.0, 0, max(0, w - crop_w)))
        y1 = int(np.clip(cy - crop_h / 2.0, 0, max(0, h - crop_h)))
        return frame[y1:y1 + crop_h, x1:x1 + crop_w]

    cropped = vc.transform(_frame_transform, apply_to=["mask"])
    # Reported size must match the cropped output so downstream resize
    # works correctly.
    cropped = cropped.with_duration(vc.duration)
    # Force dimensions: resize to exact target.  We cropped to crop_w/h which
    # already has the target aspect; resize normalises to tgt_w/tgt_h.
    out = cropped.with_effects([Resize((tgt_w, tgt_h))])
    if vc.audio is not None:
        out = out.with_audio(vc.audio)
    return out


# ------------------------------------------------------------------
# Subject-path computation
# ------------------------------------------------------------------

def _compute_subject_path(
    vc: VideoFileClip,
    crop_w: int,
    crop_h: int,
) -> list[tuple[float, float, float]]:
    """Return [(t, cx, cy), ...] — dominant-subject centre per sample.

    Times are evenly spaced across the clip.  If no subject signal is
    found we emit rule-of-thirds anchors.
    """
    cap = cv2.VideoCapture(str(vc.filename)) if hasattr(vc, "filename") and vc.filename else None

    duration = max(0.01, float(vc.duration))
    sample_ts = np.arange(0.0, duration, _SAMPLE_PERIOD).tolist()
    if not sample_ts:
        sample_ts = [duration / 2.0]

    src_w, src_h = vc.w, vc.h
    default_cx, default_cy = src_w / 2.0, src_h / 2.0

    yolo = _load_yolo()

    path: list[tuple[float, float, float]] = []
    for t in sample_ts:
        frame = _read_frame(cap, vc, t)
        if frame is None:
            path.append((float(t), default_cx, default_cy))
            continue

        cx, cy = _detect_subject(frame, yolo)
        if cx is None or cy is None:
            cx, cy = _rule_of_thirds_anchor(frame.shape[1], frame.shape[0])
        path.append((float(t), float(cx), float(cy)))

    if cap is not None:
        cap.release()

    logger.info(
        "Smart reframe path: %d samples  crop=%dx%d  first=(%.0f,%.0f)",
        len(path), crop_w, crop_h,
        path[0][1] if path else 0, path[0][2] if path else 0,
    )
    return path


def _read_frame(cap, vc, t: float) -> np.ndarray | None:
    """Read a BGR frame at time *t* from either a VideoCapture or MoviePy clip."""
    if cap is not None and cap.isOpened():
        cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000.0)
        ok, frame = cap.read()
        if ok and frame is not None:
            return frame
    # Fallback — ask MoviePy for the frame (slower but works for composites).
    try:
        rgb = vc.get_frame(min(max(t, 0.0), vc.duration - 0.01))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception:
        return None


# ------------------------------------------------------------------
# Detectors
# ------------------------------------------------------------------

def _detect_subject(frame: np.ndarray, yolo: Any) -> tuple[float | None, float | None]:
    """Return (cx, cy) of the dominant subject, or (None, None)."""
    # 1. YOLO.
    if yolo is not None:
        try:
            res = yolo.predict(frame, verbose=False, conf=0.25)
            if res:
                boxes = getattr(res[0], "boxes", None)
                if boxes is not None and len(boxes) > 0:
                    xyxy = (boxes.xyxy.cpu().numpy()
                            if hasattr(boxes.xyxy, "cpu")
                            else np.asarray(boxes.xyxy))
                    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
                    idx = int(np.argmax(areas))
                    x1, y1, x2, y2 = xyxy[idx]
                    return (float((x1 + x2) / 2.0), float((y1 + y2) / 2.0))
        except Exception:
            logger.debug("YOLO detection failed on frame — using saliency fallback",
                         exc_info=True)

    # 2. Saliency-ish fallback — gradient magnitude centroid.
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        # Keep only the top 20 % of gradient values.
        thresh = np.percentile(mag, 80)
        mask = (mag >= thresh).astype(np.float32)
        if mask.sum() > 50:
            ys, xs = np.nonzero(mask)
            cx = float(xs.mean())
            cy = float(ys.mean())
            return (cx, cy)
    except Exception:
        pass

    return (None, None)


def _rule_of_thirds_anchor(w: int, h: int) -> tuple[float, float]:
    """Return the rule-of-thirds intersection closest to frame centre."""
    candidates = [
        (w * _THIRD, h * _THIRD),
        (w * (1 - _THIRD), h * _THIRD),
        (w * _THIRD, h * (1 - _THIRD)),
        (w * (1 - _THIRD), h * (1 - _THIRD)),
    ]
    cx, cy = w / 2.0, h / 2.0
    return min(candidates, key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2)


# ------------------------------------------------------------------
# Smoothing
# ------------------------------------------------------------------

def _smoothed_centre_fn(
    path: list[tuple[float, float, float]],
    duration: float,
    src_w: int,
    src_h: int,
) -> Callable[[float], tuple[float, float]]:
    """Return ``t -> (cx, cy)`` interpolated + low-pass-smoothed."""
    if not path:
        default = (src_w / 2.0, src_h / 2.0)
        return lambda t: default

    ts = np.asarray([p[0] for p in path])
    xs = np.asarray([p[1] for p in path])
    ys = np.asarray([p[2] for p in path])

    # Moving-average smoothing.
    win = max(1, min(_SMOOTH_WINDOW, len(xs)))
    kernel = np.ones(win) / win
    xs_smooth = np.convolve(xs, kernel, mode="same")
    ys_smooth = np.convolve(ys, kernel, mode="same")

    def _fn(t: float) -> tuple[float, float]:
        t_clamped = float(min(max(t, 0.0), duration))
        cx = float(np.interp(t_clamped, ts, xs_smooth))
        cy = float(np.interp(t_clamped, ts, ys_smooth))
        return (cx, cy)

    return _fn


# ------------------------------------------------------------------
# Lazy YOLO loader (same pattern as shot_classifier; duplicated to
# avoid an import-time dependency).
# ------------------------------------------------------------------

def _load_yolo() -> Any:
    global _YOLO_MODEL, _YOLO_READY
    if _YOLO_READY is False:
        return None
    if _YOLO_MODEL is not None:
        return _YOLO_MODEL
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception:
        _YOLO_READY = False
        return None
    try:
        _YOLO_MODEL = YOLO("yolov8n.pt")
        _YOLO_READY = True
    except Exception:
        _YOLO_READY = False
        return None
    return _YOLO_MODEL
