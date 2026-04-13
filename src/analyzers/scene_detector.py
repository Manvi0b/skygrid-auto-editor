"""Analyzer that evaluates visual interest via scene-change density."""

from __future__ import annotations

import logging

from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector

from src.analyzers.base import BaseAnalyzer
from src.models.clip import Clip

logger = logging.getLogger(__name__)


class SceneDetectorAnalyzer(BaseAnalyzer):
    """Scores clips by scene-change density using PySceneDetect.

    Clips with moderate scene variety score highest; static or overly chaotic
    clips are penalised.  Also tags each clip with scene count and average
    scene length for downstream use.
    """

    # Sweet-spot: ~6–10 scene changes per minute for drone edits.
    IDEAL_SCENES_PER_MINUTE: float = 8.0

    @property
    def name(self) -> str:
        return "scene_detector"

    def analyze(self, clip: Clip) -> Clip:
        """Detect scene changes, tag with stats, and score visual variety.

        Args:
            clip: Clip to analyze.

        Returns:
            Clip with a ``scene_detector`` score (0–100), plus metadata:
            ``scene_count``, ``avg_scene_length_s``, and ``scene_boundaries``.
        """
        try:
            video = open_video(str(clip.path))
            scene_manager = SceneManager()
            scene_manager.add_detector(ContentDetector(threshold=27.0))
            scene_manager.detect_scenes(video)
            scene_list = scene_manager.get_scene_list()
        except Exception:
            logger.exception("Scene detection failed for %s", clip.path)
            return clip.with_score(self.name, 50.0)

        num_scenes = max(len(scene_list), 1)  # At least 1 (the whole clip).
        duration_min = max(clip.duration / 60.0, 0.01)
        density = num_scenes / duration_min

        # Bell-curve scoring around the ideal density (0–100 scale).
        deviation = abs(density - self.IDEAL_SCENES_PER_MINUTE)
        score = max(0.0, min(100.0, 100.0 - (deviation / self.IDEAL_SCENES_PER_MINUTE) * 100.0))
        score = round(score, 1)

        avg_scene_len = clip.duration / num_scenes

        # Scene boundary timestamps (start, end) in seconds.
        boundaries: list[tuple[float, float]] = []
        for start_tc, end_tc in scene_list:
            boundaries.append((start_tc.get_seconds(), end_tc.get_seconds()))

        result = clip.with_score(self.name, score)
        result = result.with_metadata("scene_count", num_scenes)
        result = result.with_metadata("avg_scene_length_s", round(avg_scene_len, 2))
        result = result.with_metadata("scene_boundaries", boundaries)

        if num_scenes == 1 and clip.duration > 10:
            result = result.with_tag("static")
        if density > 20:
            result = result.with_tag("chaotic")

        logger.info(
            "%s — scenes=%d  density=%.1f/min  avg_len=%.1fs  score=%s",
            clip.path.name, num_scenes, density, avg_scene_len, score,
        )
        return result
