"""Pipeline orchestrator — wires ingestion, analysis, assembly, and export."""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import logging
import random
from pathlib import Path

from src.analyzers.base import BaseAnalyzer
from src.assembler.editor import assemble
from src.assembler.edl_assembler import assemble_from_edl
from src.audio.music_sync import analyze_music
from src.config import Config
from src.export.renderer import render
from src.ingest.loader import load_clips
from src.ingest.manifest import write_project_manifest
from src.models.clip import Clip
from src.models.edl import EDL
from src.models.output_profile import OutputProfile
from src.branding.cards import apply_branding
from src.export.timeline_export import export_all as export_nle_files
from src.qc.preview import generate_contact_sheet
from src.sequencer.beat_sequencer import build_edl, build_edl_from_segments
from src.ingest.segments import extract_segments, write_segments_manifest

logger = logging.getLogger(__name__)

_BUILTIN_ANALYZER_PKG = "src.analyzers"


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def run(
    config: Config,
    output_path: Path | None = None,
    profile: OutputProfile | None = None,
) -> Path:
    """Execute the full edit pipeline end-to-end.

    Args:
        config: Validated pipeline configuration.
        output_path: Override for the final output file path.
        profile: Output profile override.

    Returns:
        Path to the rendered video file.
    """
    target = profile or config.output_profile

    # 1–4. Analyze, filter, sort (with orientation-aware loading).
    accepted = analyze_and_rank(config, profile=target)

    # Persist a project manifest for inspection / downstream tools.
    try:
        write_project_manifest(accepted, config.output_dir, config.project_name)
    except Exception:
        logger.exception("Failed to write project manifest — continuing")

    # Decide: beat-aligned (music-driven) or classic concat assembly.
    use_beats = (
        config.beat_aligned
        and config.music_track is not None
        and config.music_track.exists()
    )

    if use_beats:
        target_len = config.target_length or config.max_duration
        logger.info("Beat-aligned sequencing  target=%.1fs  pacing=%s  mood=%s",
                    target_len, config.pacing, config.mood)
        music_map = analyze_music(config.music_track)

        # If the quality_windows analyzer ran, pull best-of segments out of each
        # clip and sequence those instead of whole files.
        segments = extract_segments(accepted)
        if segments:
            try:
                write_segments_manifest(segments, config.output_dir)
            except Exception:
                logger.exception("Failed to write segments manifest — continuing")
            edl = build_edl_from_segments(
                segments=segments,
                music_map=music_map,
                target_duration=target_len,
                pacing=config.pacing,
                mood=config.mood,
            )
        else:
            edl = build_edl(
                clips=accepted,
                music_map=music_map,
                target_duration=target_len,
                pacing=config.pacing,
                mood=config.mood,
            )
        # Persist EDL alongside the output.
        try:
            _write_edl_json(edl, config.output_dir / "edl.json")
        except Exception:
            logger.exception("Failed to persist EDL — continuing")

        logger.info("\n%s", edl.summary())

        # QC contact sheet — cheap preview of cut order before the heavy encode.
        try:
            generate_contact_sheet(edl, config.output_dir / "contact_sheet.png")
        except Exception:
            logger.exception("Contact-sheet generation failed — continuing")

        # NLE handoff files (FCPXML + CMX EDL + OTIO if installed).
        try:
            export_nle_files(
                edl,
                config.output_dir,
                basename="edit",
                fps=target.fps,
                width=target.width,
                height=target.height,
                project_name=config.project_name,
            )
        except Exception:
            logger.exception("NLE export failed — continuing")

        assembled_path = assemble_from_edl(edl, config, profile=target)
    else:
        # 5. Assemble (legacy path).
        assembled_path = assemble(accepted, config, profile=target)

    # 5b. Branding — title/outro cards + lower-third overlay (v0.11.0).
    assembled_path = apply_branding(assembled_path, config, profile=target)

    # 6. Render.
    final_path = output_path or (config.output_dir / "final.mp4")
    return render(assembled_path, final_path, config, profile=target)


# ------------------------------------------------------------------
# Multi-output: analyze once, render many
# ------------------------------------------------------------------

# Aspect → default profile name (used when `outputs:` specs give aspect but no profile).
_ASPECT_PROFILE_DEFAULTS: dict[str, str] = {
    "16:9": "youtube",
    "9:16": "reels",
    "1:1": "instagram_square",
    "4k": "youtube_4k",
    "16:9@4k": "youtube_4k",
}


def run_all(
    config: Config,
    date_str: str | None = None,
) -> list[Path]:
    """Analyze clips once, then render every entry in ``config.outputs``.

    Each output spec is a dict with:

    * ``length`` (required) — target edit length in seconds.
    * ``profile`` (optional) — output profile name.
    * ``aspect`` (optional) — shorthand when no ``profile`` given;
      looked up against :data:`_ASPECT_PROFILE_DEFAULTS`.
    * ``mood`` (optional) — per-output mood override.

    Files land under ``config.output_dir/config.project_name/`` with
    the name ``{client_or_project}_{Ns}_{aspect}_{YYYYMMDD}.mp4``.

    Args:
        config: Validated pipeline configuration.
        date_str: Override the date component of the filename
            (``YYYYMMDD``).  Defaults to today.

    Returns:
        Paths of every rendered file, in output-spec order.

    Raises:
        RuntimeError: If ``config.outputs`` is empty.
    """
    from datetime import date as _date
    from src.config import resolve_output_profile

    if not config.outputs:
        raise RuntimeError("config.outputs is empty — nothing to render")

    # Expensive prep: analyze + rank once.
    base_profile = config.output_profile
    accepted = analyze_and_rank(config, profile=base_profile)

    try:
        write_project_manifest(accepted, config.output_dir, config.project_name)
    except Exception:
        logger.exception("Failed to write project manifest — continuing")

    # Pre-compute segments + music map once; they're profile-independent.
    segments = extract_segments(accepted)
    if segments:
        try:
            write_segments_manifest(segments, config.output_dir)
        except Exception:
            logger.exception("Failed to write segments manifest — continuing")

    music_map = None
    if config.music_track is not None and config.music_track.exists():
        music_map = analyze_music(config.music_track)

    date_component = date_str or _date.today().strftime("%Y%m%d")
    project_dir = config.output_dir / config.project_name
    project_dir.mkdir(parents=True, exist_ok=True)

    results: list[Path] = []
    for i, spec in enumerate(config.outputs, 1):
        length = float(spec.get("length", config.max_duration))
        profile_name = spec.get("profile") or _profile_for_aspect(spec.get("aspect"))
        if profile_name is None:
            logger.warning("Output #%d: no profile/aspect — skipping", i)
            continue
        mood_override = spec.get("mood")

        logger.info(
            "—— Rendering output %d/%d — profile=%s  length=%.1fs  mood=%s",
            i, len(config.outputs), profile_name, length, mood_override or config.mood,
        )

        per_cfg = resolve_output_profile(config, profile_name)
        if mood_override:
            per_cfg = _clone_with_mood(per_cfg, mood_override)
        per_cfg = _clone_with_target_len(per_cfg, length)

        aspect_tag = _aspect_tag(per_cfg.output_profile.aspect_ratio)
        stem_name = config.client_name or config.project_name
        filename = f"{_slug(stem_name)}_{int(length)}s_{aspect_tag}_{date_component}.mp4"
        output_path = project_dir / filename

        rendered = _render_one(per_cfg, accepted, segments, music_map, output_path)
        results.append(rendered)

    logger.info("Multi-output complete — %d file(s) written", len(results))
    return results


def _render_one(
    config: Config,
    accepted: list[Clip],
    segments: list,
    music_map,
    output_path: Path,
) -> Path:
    """Render a single output using already-analyzed clips/segments."""
    target = config.output_profile
    use_beats = (
        config.beat_aligned and music_map is not None
    )
    target_len = config.target_length or config.max_duration

    if use_beats:
        if segments:
            edl = build_edl_from_segments(
                segments=segments,
                music_map=music_map,
                target_duration=target_len,
                pacing=config.pacing,
                mood=config.mood,
            )
        else:
            edl = build_edl(
                clips=accepted,
                music_map=music_map,
                target_duration=target_len,
                pacing=config.pacing,
                mood=config.mood,
            )
        logger.info("\n%s", edl.summary())
        # Per-output NLE handoff files, named to match the rendered mp4.
        try:
            export_nle_files(
                edl,
                output_path.parent,
                basename=output_path.stem,
                fps=target.fps,
                width=target.width,
                height=target.height,
                project_name=config.project_name,
            )
        except Exception:
            logger.exception("NLE export failed — continuing")
        assembled_path = assemble_from_edl(edl, config, profile=target)
    else:
        assembled_path = assemble(accepted, config, profile=target)

    assembled_path = apply_branding(assembled_path, config, profile=target)
    return render(assembled_path, output_path, config, profile=target)


def _profile_for_aspect(aspect: str | None) -> str | None:
    """Map an ``aspect`` shorthand to a built-in profile name."""
    if not aspect:
        return None
    key = str(aspect).lower().strip()
    return _ASPECT_PROFILE_DEFAULTS.get(key)


def _aspect_tag(aspect_ratio: tuple[int, int]) -> str:
    """Render an aspect tuple as a filename-safe tag (e.g. ``"16x9"``)."""
    w, h = aspect_ratio
    return f"{w}x{h}"


def _slug(s: str) -> str:
    """Filename-safe slug of *s*."""
    out = []
    for ch in s.strip():
        if ch.isalnum():
            out.append(ch)
        elif ch in (" ", "-", "_", "."):
            out.append("_")
    return "".join(out).strip("_") or "output"


def _clone_with_mood(config: Config, mood: str) -> Config:
    """Return a copy of *config* with a new mood (without importing main.py)."""
    return _reclone(config, mood=mood)


def _clone_with_target_len(config: Config, target_length: float) -> Config:
    """Return a copy of *config* with a new target length."""
    return _reclone(config, target_length=target_length)


def _reclone(config: Config, **overrides) -> Config:
    """Minimal in-module Config cloner — keeps pipeline.py self-contained."""
    base = {f.name: getattr(config, f.name) for f in _config_fields(config)}
    base.update(overrides)
    return Config(**base)


def _config_fields(config: Config):
    """Return the Config dataclass fields (light indirection for testability)."""
    from dataclasses import fields
    return fields(config)


def _write_edl_json(edl: EDL, path: Path) -> None:
    """Serialise an EDL to JSON (small helper kept here to avoid circular imports)."""
    import json
    payload = {
        "target_duration": edl.target_duration,
        "bpm": edl.bpm,
        "mood": edl.mood,
        "pacing": edl.pacing,
        "entries": [
            {
                "source_path": str(e.source_path),
                "source_in": round(e.source_in, 3),
                "source_out": round(e.source_out, 3),
                "timeline_in": round(e.timeline_in, 3),
                "duration": round(e.duration, 3),
                "transition_in": e.transition_in,
                "transition_out": e.transition_out,
                "beat_aligned": e.beat_aligned,
                "shot_type": e.shot_type,
                "movement": e.movement,
                "score": round(e.score, 2),
                "notes": e.notes,
            }
            for e in edl.entries
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    logger.info("EDL → %s", path)


def analyze_only(
    config: Config,
    profile: OutputProfile | None = None,
) -> list[Clip]:
    """Run ingestion and analysis without assembling or rendering.

    Args:
        config: Validated pipeline configuration.
        profile: Output profile for orientation-aware sorting.

    Returns:
        All clips with scores and tags, sorted by composite score descending.
    """
    target = profile or config.output_profile
    clips = load_clips(config.input_dir, target_profile=target)
    if not clips:
        raise RuntimeError(f"No video clips found in {config.input_dir}")

    analyzers = discover_analyzers(config)
    logger.info("Active analyzers: %s", [a.name for a in analyzers])

    analyzed = _run_analyzers(clips, analyzers)
    return sorted(analyzed, key=lambda c: c.composite_score, reverse=True)


def analyze_and_rank(
    config: Config,
    profile: OutputProfile | None = None,
) -> list[Clip]:
    """Analyze, filter, and sort clips — ready for assembly.

    Args:
        config: Validated pipeline configuration.
        profile: Output profile for orientation-aware sorting.

    Returns:
        Accepted clips in assembly order.

    Raises:
        RuntimeError: If no clips survive filtering.
    """
    analyzed = analyze_only(config, profile=profile)

    accepted = [c for c in analyzed if c.composite_score >= config.min_clip_score]
    logger.info(
        "Accepted %d / %d clips (min_score=%.1f)",
        len(accepted), len(analyzed), config.min_clip_score,
    )

    if not accepted:
        raise RuntimeError(
            "All clips were filtered out — try lowering min_clip_score "
            f"(current: {config.min_clip_score})"
        )

    return _sort_clips(accepted, config.sort_by)


# ------------------------------------------------------------------
# Analyzer discovery
# ------------------------------------------------------------------

def discover_analyzers(config: Config) -> list[BaseAnalyzer]:
    """Find and instantiate all enabled analyzers from built-ins and plugins.

    Args:
        config: Pipeline configuration.

    Returns:
        Instantiated analyzer objects in ``enabled_analyzers`` order.
    """
    registry: dict[str, BaseAnalyzer] = {}

    builtin_dir = Path(__file__).resolve().parent / "analyzers"
    for py_file in sorted(builtin_dir.glob("*.py")):
        if py_file.name.startswith("_") or py_file.name == "base.py":
            continue
        module_name = f"{_BUILTIN_ANALYZER_PKG}.{py_file.stem}"
        _register_from_module(module_name, registry)

    plugins_dir = config.plugins_dir
    if plugins_dir.exists():
        for py_file in sorted(plugins_dir.glob("*.py")):
            if py_file.name.startswith("_"):
                continue
            spec = importlib.util.spec_from_file_location(
                f"plugins.{py_file.stem}", py_file,
            )
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                try:
                    spec.loader.exec_module(module)  # type: ignore[arg-type]
                except Exception:
                    logger.exception("Failed to load plugin %s", py_file)
                    continue
                for attr_name in dir(module):
                    cls = getattr(module, attr_name)
                    if _is_analyzer_class(cls):
                        instance = cls()
                        registry[instance.name] = instance

    enabled: list[BaseAnalyzer] = []
    for name in config.enabled_analyzers:
        if name in registry:
            enabled.append(registry[name])
        else:
            logger.warning("Analyzer '%s' is enabled but was not found", name)
    return enabled


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _run_analyzers(
    clips: list[Clip],
    analyzers: list[BaseAnalyzer],
) -> list[Clip]:
    """Run every analyzer on every clip."""
    analyzed: list[Clip] = []
    for clip in clips:
        for analyzer in analyzers:
            try:
                clip = analyzer.analyze(clip)
            except Exception:
                logger.exception(
                    "Analyzer %s failed on %s — skipping analyzer",
                    analyzer.name, clip.path.name,
                )
        analyzed.append(clip)
    return analyzed


def _register_from_module(
    module_name: str,
    registry: dict[str, BaseAnalyzer],
) -> None:
    """Import *module_name* and register concrete BaseAnalyzer subclasses."""
    try:
        module = importlib.import_module(module_name)
    except Exception:
        logger.exception("Failed to import %s", module_name)
        return

    for attr_name in dir(module):
        cls = getattr(module, attr_name)
        if _is_analyzer_class(cls):
            instance = cls()
            registry[instance.name] = instance


def _is_analyzer_class(obj: object) -> bool:
    """Return True if *obj* is a concrete BaseAnalyzer subclass."""
    return (
        inspect.isclass(obj)
        and issubclass(obj, BaseAnalyzer)
        and obj is not BaseAnalyzer
        and not inspect.isabstract(obj)
    )


def _sort_clips(clips: list[Clip], strategy: str) -> list[Clip]:
    """Sort clips according to the named strategy."""
    match strategy:
        case "score":
            return sorted(clips, key=lambda c: c.composite_score, reverse=True)
        case "chronological":
            return sorted(clips, key=lambda c: c.path.name)
        case "random":
            shuffled = list(clips)
            random.shuffle(shuffled)
            return shuffled
        case _:
            logger.warning("Unknown sort strategy '%s' — defaulting to score", strategy)
            return sorted(clips, key=lambda c: c.composite_score, reverse=True)
