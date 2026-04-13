"""Analyzer that detects camera shake / instability in video clips."""

from __future__ import annotations

import logging

import cv2
import numpy as np

from src.analyzers.base import BaseAnalyzer
from src.models.clip import Clip

logger = logging.getLogger(__name__)

# Lucas-Kanade optical-flow parameters.
_LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)

# Shi-Tomasi corner detection parameters.
_FEATURE_PARAMS = dict(
    maxCorners=200,
    qualityLevel=0.01,
    minDistance=30,
    blockSize=7,
)

# Maximum number of frames to sample (avoids processing hour-long clips).
_MAX_SAMPLE_FRAMES = 900


class ShakeDetector(BaseAnalyzer):
    """Scores clips by camera stability using Lucas-Kanade optical flow.

    Analyses inter-frame motion vectors to estimate camera shake.
    A perfectly stable clip scores 100; heavy shake approaches 0.
    """

    @property
    def name(self) -> str:
        return "shake_detector"

    def analyze(self, clip: Clip) -> Clip:
        """Compute a stability score for *clip* via sparse optical flow.

        Args:
            clip: Clip to analyze.

        Returns:
            Clip with a ``shake_detector`` score (0–100), motion metadata,
            and a ``"shaky"`` tag if the score is below 40.
        """
        cap = cv2.VideoCapture(str(clip.path))
        if not cap.isOpened():
            logger.warning("Cannot open %s — assigning neutral score", clip.path)
            return clip.with_score(self.name, 50.0)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Sample every Nth frame for long clips.
        step = max(1, total_frames // _MAX_SAMPLE_FRAMES)

        ret, prev_frame = cap.read()
        if not ret:
            cap.release()
            return clip.with_score(self.name, 50.0)

        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        motion_magnitudes: list[float] = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx % step != 0:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect features in previous frame.
            features = cv2.goodFeaturesToTrack(prev_gray, **_FEATURE_PARAMS)
            if features is None or len(features) < 4:
                prev_gray = gray
                continue

            # Track features forward with optical flow.
            tracked, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, gray, features, None, **_LK_PARAMS,
            )
            if tracked is None:
                prev_gray = gray
                continue

            # Keep only successfully tracked points.
            good_mask = status.ravel() == 1
            prev_pts = features[good_mask].reshape(-1, 2)
            curr_pts = tracked[good_mask].reshape(-1, 2)

            if len(prev_pts) < 4:
                prev_gray = gray
                continue

            # Compute per-point displacement magnitudes.
            displacements = curr_pts - prev_pts
            magnitudes = np.sqrt(
                displacements[:, 0] ** 2 + displacements[:, 1] ** 2
            )
            motion_magnitudes.append(float(np.median(magnitudes)))
            prev_gray = gray

        cap.release()

        if not motion_magnitudes:
            return clip.with_score(self.name, 50.0)

        avg_motion = float(np.mean(motion_magnitudes))
        max_motion = float(np.max(motion_magnitudes))
        std_motion = float(np.std(motion_magnitudes))

        # Score: low average motion → high stability.
        # Typical drone footage has avg_motion 1–8 px/frame.
        # >15 px/frame is very shaky.
        score = max(0.0, min(100.0, 100.0 - (avg_motion * 6.0)))
        # Penalise high variance (jerky motion) on top of average.
        score = max(0.0, score - std_motion * 2.0)
        score = round(score, 1)

        result = clip.with_score(self.name, score)
        result = result.with_metadata("avg_motion_px", round(avg_motion, 2))
        result = result.with_metadata("max_motion_px", round(max_motion, 2))
        result = result.with_metadata("std_motion_px", round(std_motion, 2))

        if score < 40:
            result = result.with_tag("shaky")
        if score >= 85:
            result = result.with_tag("stable")

        logger.info(
            "%s — stability=%s  avg_motion=%.2f px",
            clip.path.name, score, avg_motion,
        )
        return result
