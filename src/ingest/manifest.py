"""Project manifest — persistent record of the ingest stage.

Writes ``project.json`` inside the project output directory so that
subsequent stages (and human inspection) can see what was ingested,
what metadata was extracted, and what proxies exist.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from src.models.clip import Clip

logger = logging.getLogger(__name__)


def write_project_manifest(
    clips: list[Clip],
    output_dir: Path,
    project_name: str = "project",
) -> Path:
    """Serialise *clips* to ``<output_dir>/project.json``.

    Args:
        clips: The full ingested clip list (after analysis is fine too).
        output_dir: Project output directory.
        project_name: Label stored inside the manifest.

    Returns:
        Path to the written manifest file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "project.json"

    payload = {
        "project_name": project_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "clip_count": len(clips),
        "clips": [_clip_to_dict(c) for c in clips],
    }

    with open(manifest_path, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)

    logger.info("Project manifest → %s  (%d clips)", manifest_path, len(clips))
    return manifest_path


def _clip_to_dict(c: Clip) -> dict:
    """Convert a Clip to a JSON-serialisable dict."""
    return {
        "path": str(c.path),
        "duration": round(c.duration, 3),
        "width": c.width,
        "height": c.height,
        "fps": round(c.fps, 3),
        "orientation": c.orientation,
        "source_profile": c.source_profile,
        "composite_score": round(c.composite_score, 2),
        "scores": {k: round(v, 2) for k, v in c.scores.items()},
        "tags": list(c.tags),
        "metadata": {k: _jsonify(v) for k, v in c.metadata.items()},
    }


def _jsonify(v):
    """Coerce values to JSON-safe types."""
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [_jsonify(x) for x in v]
    if isinstance(v, dict):
        return {k: _jsonify(x) for k, x in v.items()}
    return str(v)
