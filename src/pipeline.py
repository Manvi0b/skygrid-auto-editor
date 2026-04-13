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
from src.config import Config
from src.export.renderer import render
from src.ingest.loader import load_clips
from src.models.clip import Clip

logger = logging.getLogger(__name__)

_BUILTIN_ANALYZER_PKG = "src.analyzers"


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def run(config: Config, output_path: Path | None = None) -> Path:
    """Execute the full edit pipeline end-to-end.

    Steps:
        1. Ingest clips from the configured input directory.
        2. Discover and instantiate enabled analyzers (built-in + plugins).
        3. Run each analyzer over every clip.
        4. Filter and sort clips by composite score.
        5. Assemble the timeline with transitions.
        6. Render the final output (optionally mixing in music).

    Args:
        config: Validated pipeline configuration.
        output_path: Override for the final output file path.

    Returns:
        Path to the rendered video file.
    """
    # 1–4. Analyze and select.
    accepted = analyze_and_rank(config)

    # 5. Assemble.
    assembled_path = assemble(accepted, config)

    # 6. Render.
    final_path = output_path or (config.output_dir / "final.mp4")
    return render(assembled_path, final_path, config)


def analyze_only(config: Config) -> list[Clip]:
    """Run ingestion and analysis without assembling or rendering.

    Args:
        config: Validated pipeline configuration.

    Returns:
        All clips with analysis scores and tags attached, sorted by
        composite score descending.
    """
    clips = load_clips(config.input_dir)
    if not clips:
        raise RuntimeError(f"No video clips found in {config.input_dir}")

    analyzers = discover_analyzers(config)
    logger.info("Active analyzers: %s", [a.name for a in analyzers])

    analyzed = _run_analyzers(clips, analyzers)
    return sorted(analyzed, key=lambda c: c.composite_score, reverse=True)


def analyze_and_rank(config: Config) -> list[Clip]:
    """Analyze, filter, and sort clips — ready for assembly.

    Args:
        config: Validated pipeline configuration.

    Returns:
        Accepted clips in assembly order.

    Raises:
        RuntimeError: If no clips survive filtering.
    """
    analyzed = analyze_only(config)

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

    Discovery order:
        1. Built-in analyzers in ``src/analyzers/``.
        2. Plugin analyzers in the configured ``plugins/`` directory.

    Only analyzers whose ``name`` appears in ``config.enabled_analyzers``
    are returned.

    Args:
        config: Pipeline configuration.

    Returns:
        Instantiated analyzer objects in ``enabled_analyzers`` order.
    """
    registry: dict[str, BaseAnalyzer] = {}

    # Built-in modules.
    builtin_dir = Path(__file__).resolve().parent / "analyzers"
    for py_file in sorted(builtin_dir.glob("*.py")):
        if py_file.name.startswith("_") or py_file.name == "base.py":
            continue
        module_name = f"{_BUILTIN_ANALYZER_PKG}.{py_file.stem}"
        _register_from_module(module_name, registry)

    # Plugin modules.
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
    """Run every analyzer on every clip, collecting results.

    Args:
        clips: Raw clips from the loader.
        analyzers: Instantiated analyzers to apply.

    Returns:
        Clips enriched with scores, metadata, and tags.
    """
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
    """Import *module_name* and register any concrete BaseAnalyzer subclasses."""
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
    """Sort clips according to the named strategy.

    Args:
        clips: Clips to sort.
        strategy: ``"score"``, ``"chronological"``, or ``"random"``.

    Returns:
        Sorted (or shuffled) list of clips.
    """
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
