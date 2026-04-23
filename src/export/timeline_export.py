"""Timeline export — FCPXML and CMX 3600 EDL writers.

Writes an :class:`EDL` to a format an NLE can open so the
auto-generated edit is never a black box.  Supported:

* **FCPXML v1.9** — opens in Premiere Pro, DaVinci Resolve, Final
  Cut Pro.  Produced directly without any external dependency.
* **CMX 3600 EDL** — the old-school text format, still accepted by
  every NLE for quick conforming.
* **OTIO** — OpenTimelineIO interchange, when the ``opentimelineio``
  package is available.  Otherwise silently skipped.

All three honour the source in/out points from the EDL so the NLE
timeline lands on the same frames the auto-edit did.
"""

from __future__ import annotations

import logging
from pathlib import Path
from xml.etree import ElementTree as ET

from src.models.edl import EDL

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# FCPXML
# ------------------------------------------------------------------

def write_fcpxml(
    edl: EDL,
    output_path: Path,
    fps: int = 30,
    width: int = 1920,
    height: int = 1080,
    project_name: str = "SkyGrid Edit",
) -> Path:
    """Write *edl* as an FCPXML 1.9 file openable in Premiere / Resolve / FCP.

    Args:
        edl: Edit Decision List to serialise.
        output_path: Destination ``.fcpxml`` file.
        fps: Sequence frame rate.
        width: Sequence pixel width.
        height: Sequence pixel height.
        project_name: Name shown in the NLE project browser.

    Returns:
        Path to the written file.
    """
    fcpxml = ET.Element("fcpxml", {"version": "1.9"})

    resources = ET.SubElement(fcpxml, "resources")
    fmt_id = "r0"
    ET.SubElement(resources, "format", {
        "id": fmt_id,
        "name": f"SG_{width}x{height}_{fps}p",
        "frameDuration": f"1/{fps}s",
        "width": str(width),
        "height": str(height),
    })

    # Dedupe source files → asset records.
    asset_ids: dict[str, str] = {}
    next_asset = 1
    for entry in edl.entries:
        key = str(entry.source_path)
        if key not in asset_ids:
            asset_ids[key] = f"a{next_asset}"
            next_asset += 1

    for src_path, aid in asset_ids.items():
        name = Path(src_path).stem
        ET.SubElement(resources, "asset", {
            "id": aid,
            "name": name,
            "src": Path(src_path).absolute().as_uri(),
            "hasVideo": "1",
            "hasAudio": "1",
            "format": fmt_id,
        })

    library = ET.SubElement(fcpxml, "library")
    event = ET.SubElement(library, "event", {"name": project_name})
    project = ET.SubElement(event, "project", {"name": project_name})
    sequence = ET.SubElement(project, "sequence", {
        "format": fmt_id,
        "duration": _secs(edl.total_duration or edl.target_duration),
    })
    spine = ET.SubElement(sequence, "spine")

    for entry in edl.entries:
        ET.SubElement(spine, "asset-clip", {
            "ref": asset_ids[str(entry.source_path)],
            "name": entry.source_path.stem,
            "offset": _secs(entry.timeline_in),
            "start": _secs(entry.source_in),
            "duration": _secs(entry.duration),
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(fcpxml)
    ET.indent(tree, space="  ")
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    logger.info("FCPXML → %s (%d cuts)", output_path, len(edl.entries))
    return output_path


def _secs(seconds: float) -> str:
    """Format *seconds* as an FCPXML rational duration (e.g. ``"5/1s"``)."""
    # FCPXML accepts integer-numerator rationals — 300/30 = 10s at 30fps.
    # Simpler: emit "N/1000s" for millisecond precision.
    ms = int(round(seconds * 1000))
    return f"{ms}/1000s"


# ------------------------------------------------------------------
# CMX 3600 EDL
# ------------------------------------------------------------------

def write_cmx_edl(
    edl: EDL,
    output_path: Path,
    fps: int = 30,
    title: str = "SkyGrid Edit",
) -> Path:
    """Write *edl* as a CMX 3600 text EDL.

    Args:
        edl: EDL to serialise.
        output_path: Destination ``.edl`` file.
        fps: Frame rate used for timecode conversion.
        title: Title line at the top of the EDL.

    Returns:
        Path to the written file.
    """
    lines: list[str] = [f"TITLE: {title}", "FCM: NON-DROP FRAME", ""]

    for i, entry in enumerate(edl.entries, 1):
        src_in_tc = _timecode(entry.source_in, fps)
        src_out_tc = _timecode(entry.source_out, fps)
        tl_in_tc = _timecode(entry.timeline_in, fps)
        tl_out_tc = _timecode(entry.timeline_out, fps)
        # Format: nnn  REEL     V/A     C        src_in src_out tl_in tl_out
        lines.append(
            f"{i:03d}  AX       V     C        "
            f"{src_in_tc} {src_out_tc} {tl_in_tc} {tl_out_tc}"
        )
        lines.append(f"* FROM CLIP NAME: {entry.source_path.name}")
        if entry.notes:
            lines.append(f"* COMMENT: {entry.notes}")
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("CMX EDL → %s (%d cuts)", output_path, len(edl.entries))
    return output_path


def _timecode(seconds: float, fps: int) -> str:
    """Convert seconds to HH:MM:SS:FF timecode (non-drop)."""
    total_frames = int(round(seconds * fps))
    ff = total_frames % fps
    total_sec = total_frames // fps
    ss = total_sec % 60
    mm = (total_sec // 60) % 60
    hh = total_sec // 3600
    return f"{hh:02d}:{mm:02d}:{ss:02d}:{ff:02d}"


# ------------------------------------------------------------------
# OpenTimelineIO (optional)
# ------------------------------------------------------------------

def write_otio(
    edl: EDL,
    output_path: Path,
    fps: int = 30,
) -> Path | None:
    """Write *edl* as an OpenTimelineIO JSON file, if ``opentimelineio`` is installed.

    Args:
        edl: EDL to serialise.
        output_path: Destination ``.otio`` file.
        fps: Frame rate used when populating range rationals.

    Returns:
        Path on success, or ``None`` when ``opentimelineio`` is not installed.
    """
    try:
        import opentimelineio as otio  # type: ignore
    except Exception:
        logger.info("opentimelineio not installed — skipping .otio export")
        return None

    timeline = otio.schema.Timeline(name="SkyGrid Edit")
    track = otio.schema.Track(kind="Video")
    timeline.tracks.append(track)

    for entry in edl.entries:
        media = otio.schema.ExternalReference(
            target_url=Path(entry.source_path).absolute().as_uri(),
        )
        clip = otio.schema.Clip(
            name=entry.source_path.stem,
            media_reference=media,
            source_range=otio.opentime.TimeRange(
                start_time=otio.opentime.RationalTime(entry.source_in * fps, fps),
                duration=otio.opentime.RationalTime(entry.duration * fps, fps),
            ),
        )
        track.append(clip)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    otio.adapters.write_to_file(timeline, str(output_path))
    logger.info("OTIO → %s (%d cuts)", output_path, len(edl.entries))
    return output_path


# ------------------------------------------------------------------
# Convenience: write all supported formats at once
# ------------------------------------------------------------------

def export_all(
    edl: EDL,
    output_dir: Path,
    basename: str = "edit",
    fps: int = 30,
    width: int = 1920,
    height: int = 1080,
    project_name: str = "SkyGrid Edit",
) -> list[Path]:
    """Write FCPXML + CMX EDL (+ OTIO if available) under *output_dir*.

    Args:
        edl: EDL to export.
        output_dir: Destination directory.
        basename: Stem used for each file.
        fps: Sequence frame rate.
        width: Sequence pixel width.
        height: Sequence pixel height.
        project_name: NLE project/event name.

    Returns:
        List of paths that were written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    written.append(write_fcpxml(
        edl, output_dir / f"{basename}.fcpxml",
        fps=fps, width=width, height=height, project_name=project_name,
    ))
    written.append(write_cmx_edl(edl, output_dir / f"{basename}.edl", fps=fps))
    otio_path = write_otio(edl, output_dir / f"{basename}.otio", fps=fps)
    if otio_path is not None:
        written.append(otio_path)
    return written
