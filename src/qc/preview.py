"""QC preview utilities — contact-sheet PNG + low-res proxy render.

Used between sequencing and final render so the operator can sanity-check
the cut order without waiting for a full encode.
"""

from __future__ import annotations

import logging
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np

from src.models.edl import EDL

logger = logging.getLogger(__name__)


def generate_contact_sheet(
    edl: EDL,
    output_path: Path,
    tile_width: int = 320,
    cols: int = 4,
) -> Path:
    """Render a contact-sheet PNG with one thumbnail per EDL cut.

    Args:
        edl: EDL to visualise.
        output_path: Destination PNG file.
        tile_width: Width of each thumbnail in pixels.
        cols: Number of columns in the grid.

    Returns:
        Path to the written PNG.
    """
    if not edl.entries:
        raise ValueError("EDL is empty — nothing to preview")

    tiles: list[np.ndarray] = []
    for i, entry in enumerate(edl.entries, 1):
        thumb = _extract_thumbnail(entry.source_path, entry.source_in, tile_width)
        if thumb is None:
            thumb = _placeholder_tile(tile_width, int(tile_width * 9 / 16))
        thumb = _annotate_tile(
            thumb,
            index=i,
            timeline_in=entry.timeline_in,
            duration=entry.duration,
            notes=entry.notes,
            score=entry.score,
        )
        tiles.append(thumb)

    grid = _compose_grid(tiles, cols=cols)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), grid)
    logger.info("Contact sheet → %s  (%d tiles)", output_path, len(tiles))
    return output_path


def generate_low_res_proxy(
    assembled_path: Path,
    output_path: Path,
    height: int = 480,
    max_duration: float = 30.0,
) -> Path:
    """Render a small, fast proxy preview of an assembled timeline.

    Args:
        assembled_path: Full-resolution intermediate video.
        output_path: Destination proxy file (e.g. ``preview.mp4``).
        height: Proxy height in pixels (width scales to preserve aspect).
        max_duration: Cap the proxy at this many seconds.

    Returns:
        Path to the written proxy file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-t", str(max_duration), "-i", str(assembled_path),
        "-vf", f"scale=-2:{height}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
        "-c:a", "aac", "-b:a", "96k",
        "-movflags", "+faststart",
        str(output_path),
    ]
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    logger.info("Proxy preview → %s", output_path)
    return output_path


# ------------------------------------------------------------------
# Internals
# ------------------------------------------------------------------

def _extract_thumbnail(source: Path, t: float, width: int) -> np.ndarray | None:
    """Grab a frame at time *t* from *source* and resize to *width*."""
    cap = cv2.VideoCapture(str(source))
    try:
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
        h, w = frame.shape[:2]
        scale = width / max(w, 1)
        return cv2.resize(frame, (width, int(h * scale)))
    finally:
        cap.release()


def _placeholder_tile(w: int, h: int) -> np.ndarray:
    """Grey placeholder when a thumbnail can't be extracted."""
    img = np.full((h, w, 3), 40, dtype=np.uint8)
    cv2.putText(img, "no preview", (10, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    return img


def _annotate_tile(
    tile: np.ndarray,
    index: int,
    timeline_in: float,
    duration: float,
    notes: str,
    score: float,
) -> np.ndarray:
    """Overlay cut metadata on a thumbnail."""
    h, w = tile.shape[:2]
    bar_h = 22
    canvas = np.zeros((h + bar_h, w, 3), dtype=np.uint8)
    canvas[:h] = tile
    # Bottom status bar.
    cv2.rectangle(canvas, (0, h), (w, h + bar_h), (25, 25, 25), -1)
    text = f"#{index}  t={timeline_in:0.1f}s  {duration:0.1f}s  s={score:0.0f}  {notes}"
    cv2.putText(canvas, text, (6, h + bar_h - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (240, 240, 240), 1, cv2.LINE_AA)
    return canvas


def _compose_grid(tiles: list[np.ndarray], cols: int) -> np.ndarray:
    """Arrange *tiles* into a rows×cols grid with uniform sizing."""
    # Uniform tile size: use the max per dimension.
    max_h = max(t.shape[0] for t in tiles)
    max_w = max(t.shape[1] for t in tiles)

    padded: list[np.ndarray] = []
    for t in tiles:
        pad = np.zeros((max_h, max_w, 3), dtype=np.uint8)
        h, w = t.shape[:2]
        pad[:h, :w] = t
        padded.append(pad)

    rows = math.ceil(len(padded) / cols)
    grid = np.zeros((rows * max_h, cols * max_w, 3), dtype=np.uint8)
    for i, tile in enumerate(padded):
        r, c = divmod(i, cols)
        grid[r * max_h:(r + 1) * max_h, c * max_w:(c + 1) * max_w] = tile
    return grid
