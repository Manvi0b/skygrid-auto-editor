"""Per-clip preprocessing — motion stabilization + horizon auto-level.

Runs before the assembler.  For each accepted clip, applies (optionally)
stabilization and a rotation that cancels the detected horizon tilt,
writes a cached preprocessed file under ``<output_dir>/_preproc/``, and
returns new Clip objects pointing at the cached files.

Design choices
--------------
* **Stabilization** uses ffmpeg's built-in ``deshake`` filter.  It's a
  single-pass crop-and-rematch — not as strong as ``vidstab`` but it
  ships with every ffmpeg build, so there's no extra dependency.
  ``vidstab`` is preferred when the binary has it compiled in; we
  detect that once and fall back.
* **Levelling** uses a single rotation applied after ``deshake``, so
  the crop absorbs the rotation's black corners.  We over-scale by
  10 % to hide edge artefacts.
* Results are cached — the preprocessed output is keyed on
  ``(path, mtime, stabilize, tilt)`` so re-runs skip the work.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

from src.config import Config
from src.models.clip import Clip
from src.preprocess.level import detect_horizon_tilt

logger = logging.getLogger(__name__)

# Detect vidstab once per process.
_VIDSTAB_AVAILABLE: bool | None = None


def _has_vidstab() -> bool:
    """Return True if this ffmpeg binary ships with ``vidstab``."""
    global _VIDSTAB_AVAILABLE
    if _VIDSTAB_AVAILABLE is not None:
        return _VIDSTAB_AVAILABLE
    try:
        out = subprocess.run(
            ["ffmpeg", "-hide_banner", "-filters"],
            capture_output=True, text=True, check=True,
        ).stdout
        _VIDSTAB_AVAILABLE = "vidstabdetect" in out and "vidstabtransform" in out
    except Exception:
        _VIDSTAB_AVAILABLE = False
    return _VIDSTAB_AVAILABLE


def preprocess_clips(clips: list[Clip], config: Config) -> list[Clip]:
    """Return clones of *clips* with stabilization + horizon level applied.

    No-op (returns the input unchanged) when both knobs are off in
    ``config.preprocessing``.

    Args:
        clips: Accepted clips from the analyzer stage.
        config: Pipeline configuration.

    Returns:
        A new list of Clip objects — same length, same order, but with
        :attr:`Clip.path` pointing at the preprocessed cache files where
        preprocessing was needed.
    """
    p = getattr(config, "preprocessing", {}) or {}
    stabilize = bool(p.get("stabilize", False))
    level = bool(p.get("level_horizon", False))
    if not (stabilize or level):
        return clips

    max_tilt = float(p.get("max_tilt_deg", 6.0))
    min_tilt = float(p.get("min_tilt_deg", 0.5))
    overscale = float(p.get("overscale", 1.08))
    deshake_rx = int(p.get("deshake_rx", 32))
    deshake_ry = int(p.get("deshake_ry", 32))

    cache_dir = config.output_dir / "_preproc"
    cache_dir.mkdir(parents=True, exist_ok=True)

    out: list[Clip] = []
    for clip in clips:
        tilt = detect_horizon_tilt(clip.path) if level else 0.0
        if abs(tilt) < min_tilt:
            tilt = 0.0
        if abs(tilt) > max_tilt:
            logger.info("Clamping extreme tilt %.1f° → %.1f° on %s",
                        tilt, max_tilt, clip.path.name)
            tilt = math.copysign(max_tilt, tilt)

        if not stabilize and tilt == 0.0:
            out.append(clip)
            continue

        cache_path = _cache_path(cache_dir, clip.path, stabilize, tilt)
        if not cache_path.exists():
            _render_preproc(
                clip.path, cache_path,
                stabilize=stabilize, tilt_deg=tilt,
                overscale=overscale, rx=deshake_rx, ry=deshake_ry,
            )
        else:
            logger.debug("Preproc cache hit: %s", cache_path.name)

        new_meta = dict(clip.metadata)
        new_meta["preproc"] = {
            "stabilized": stabilize,
            "horizon_tilt_deg": tilt,
            "engine": "vidstab" if (stabilize and _has_vidstab()) else
                      ("deshake" if stabilize else None),
        }
        out.append(replace(clip, path=cache_path, metadata=new_meta))

    return out


# ------------------------------------------------------------------
# Internals
# ------------------------------------------------------------------

def _cache_path(cache_dir: Path, src: Path, stabilize: bool, tilt: float) -> Path:
    """Deterministic cache filename for a (src, params) tuple."""
    mtime = src.stat().st_mtime if src.exists() else 0
    key = json.dumps({
        "src": str(src.resolve()),
        "mtime": mtime,
        "stabilize": stabilize,
        "tilt": round(tilt, 3),
    }, sort_keys=True)
    digest = hashlib.sha1(key.encode()).hexdigest()[:10]
    return cache_dir / f"{src.stem}__{digest}.mp4"


def _render_preproc(
    src: Path,
    dst: Path,
    *,
    stabilize: bool,
    tilt_deg: float,
    overscale: float,
    rx: int,
    ry: int,
) -> None:
    """Render the preprocessed copy using ffmpeg.

    Stabilization path prefers ``vidstab`` (two-pass) when available,
    otherwise falls back to single-pass ``deshake``.  The horizon
    rotation is always applied last so the crop absorbs black corners.
    """
    rot = math.radians(-tilt_deg)  # counter-rotate to level
    vf_tail = ""
    if tilt_deg != 0.0:
        # Over-scale + rotate + centre-crop back to original aspect.
        # `ow*overscale` expands canvas; `rotate` fills outside with black,
        # which we then crop to the input size.
        vf_tail = (
            f",scale=iw*{overscale}:ih*{overscale}"
            f",rotate={rot}:c=black:ow=rotw({rot}):oh=roth({rot})"
            f",crop=iw/{overscale}:ih/{overscale}"
        )

    if stabilize and _has_vidstab():
        transforms = dst.with_suffix(".trf")
        # Pass 1: analysis
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(src),
             "-vf", f"vidstabdetect=shakiness=5:accuracy=15:result={transforms}",
             "-f", "null", "-"],
            capture_output=True, check=True,
        )
        # Pass 2: transform + optional rotation
        vf = (
            f"vidstabtransform=input={transforms}:"
            f"zoom=1:smoothing=20:interpol=bicubic{vf_tail}"
            f",unsharp=5:5:0.8:3:3:0.4"
        )
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(src),
             "-vf", vf,
             "-c:v", "libx264", "-preset", "medium", "-crf", "18",
             "-c:a", "copy",
             str(dst)],
            capture_output=True, check=True,
        )
        try:
            Path(transforms).unlink(missing_ok=True)
        except Exception:
            pass
        logger.info("vidstab + level(%.1f°) → %s", tilt_deg, dst.name)
        return

    # Fallback: deshake (always available).
    deshake = f"deshake=rx={rx}:ry={ry}"
    vf = f"{deshake}{vf_tail}" if stabilize else vf_tail.lstrip(",")
    if not vf:
        # No-op: just copy.
        shutil.copy(src, dst)
        return
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(src),
             "-vf", vf,
             "-c:v", "libx264", "-preset", "medium", "-crf", "18",
             "-c:a", "copy",
             str(dst)],
            capture_output=True, text=True, check=True,
        )
    except subprocess.CalledProcessError as exc:
        logger.error("Preprocessing failed on %s — falling back to original.\n%s",
                     src.name, (exc.stderr or "")[-600:])
        shutil.copy(src, dst)
        return
    logger.info("deshake + level(%.1f°) → %s", tilt_deg, dst.name)
