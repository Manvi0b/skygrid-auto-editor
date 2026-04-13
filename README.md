# SkyGrid Auto Editor

A modular Python video editing pipeline built for drone footage. Drop clips into a folder, run a single command, and get a polished edit — scored, trimmed, transitioned, and rendered automatically.

## What This Project Does

SkyGrid Auto Editor ingests raw video clips, runs them through a plugin-based analysis system to score quality, filters out the bad takes, trims dead air, applies transitions, and renders a final video — all from the command line.

### Pipeline Flow

```
Input Clips → Ingest → Analyze → Rank/Filter → Assemble → Render → Output
```

1. **Ingest** — Scans an input folder for video files (.mp4, .mov, .avi, .mkv, .webm, .m4v, .mts) and extracts metadata (duration, resolution, fps) via ffprobe.
2. **Analyze** — Runs each clip through a set of analyzers that score quality on a 0–100 scale.
3. **Rank & Filter** — Sorts clips by composite score and drops anything below the configured minimum threshold.
4. **Assemble** — Trims leading/trailing silence from each clip, applies crossfade or fade-to-black transitions, and concatenates into a timeline respecting a max duration budget.
5. **Render** — Re-encodes the final video via FFmpeg with configurable resolution, codec, bitrate, and preset. Optionally mixes in a background music track.

## Built-in Analyzers

| Analyzer | What It Measures | How It Scores |
|---|---|---|
| **Shake Detector** | Camera stability via Lucas-Kanade optical flow. Tracks feature points frame-to-frame and measures displacement magnitudes. | Low motion = high score. Penalises jerk (high variance). Tags clips `"shaky"` (<40) or `"stable"` (≥85). |
| **Scene Detector** | Visual variety via PySceneDetect content-aware scene boundaries. | Bell-curve scoring around ~8 scene changes/minute. Tags `"static"` (single scene, >10s) or `"chaotic"` (>20/min). Stores scene count, average scene length, and boundary timestamps. |
| **Audio Analyzer** | Audio presence, RMS/peak volume in dB, and silence segments. | Peaks at -20 dB RMS (well-recorded outdoor audio). Penalises excessive silence. Tags `"no_audio"`, `"silent"`, `"quiet"`, or `"loud"`. Detects and records individual silence segments ≥0.5s. |

## Plugin System

Any `.py` file dropped into the `plugins/` directory is auto-discovered at runtime. To create a custom analyzer:

```python
from src.analyzers.base import BaseAnalyzer
from src.models.clip import Clip

class MyCustomAnalyzer(BaseAnalyzer):
    @property
    def name(self) -> str:
        return "my_custom"

    def analyze(self, clip: Clip) -> Clip:
        score = do_something(clip.path)
        return clip.with_score(self.name, score)
```

Then enable it in `config.yaml`:

```yaml
pipeline:
  enabled_analyzers:
    - "shake_detector"
    - "scene_detector"
    - "audio_analyzer"
    - "my_custom"
```

## Project Structure

```
skygrid-auto-editor/
├── src/
│   ├── main.py              # CLI entry point (Click) — edit, analyze, list
│   ├── pipeline.py          # Orchestrates the full pipeline + plugin discovery
│   ├── config.py            # Loads settings from config.yaml
│   ├── models/
│   │   └── clip.py          # Immutable Clip dataclass (path, scores, metadata, tags)
│   ├── ingest/
│   │   └── loader.py        # Scans input folder, probes video metadata via ffprobe
│   ├── analyzers/
│   │   ├── base.py          # Abstract BaseAnalyzer class
│   │   ├── shake_detector.py  # Optical flow camera stability scoring
│   │   ├── scene_detector.py  # PySceneDetect visual variety scoring
│   │   └── audio_analyzer.py  # Volume, silence, and audio presence analysis
│   ├── assembler/
│   │   └── editor.py        # Dead-air trimming, transitions, timeline assembly
│   ├── audio/
│   │   └── music_sync.py    # Beat detection, BPM estimation, cut-to-beat snapping
│   ├── export/
│   │   └── renderer.py      # FFmpeg wrapper for final render + music mixing
│   └── utils/
│       └── ffmpeg_helpers.py # ffprobe/ffmpeg utilities
├── plugins/                  # Drop custom analyzers here (auto-discovered)
├── input/                    # Place source clips here
├── output/                   # Rendered output goes here
├── config.yaml               # All pipeline settings
└── requirements.txt
```

## Installation

Requires Python 3.11+ and FFmpeg installed on your system.

```bash
git clone https://github.com/Manvi0b/skygrid-auto-editor.git
cd skygrid-auto-editor
pip install -r requirements.txt
```

## Usage

### Full edit pipeline

```bash
python -m src.main edit --input ./input --output ./output/final.mp4
```

### Analyze clips without rendering

```bash
python -m src.main analyze --input ./input
```

Prints a detailed report for each clip: scores from every analyzer, tags, resolution, duration, and key metadata like RMS volume and motion magnitude.

### List ranked clips

```bash
python -m src.main list --input ./input
python -m src.main list --input ./input --top 5
```

Outputs a summary table sorted by composite score.

### Options

```bash
python -m src.main --help           # Show all commands
python -m src.main edit --help      # Show edit options
python -m src.main -v edit ...      # Enable debug logging
python -m src.main --config custom.yaml edit ...  # Use a custom config file
```

## Configuration

All settings live in `config.yaml`:

```yaml
pipeline:
  max_duration: 120          # Max output length in seconds
  min_clip_score: 30         # Minimum score (0–100) to include a clip
  sort_by: "score"           # "score", "chronological", or "random"

transitions:
  style: "crossfade"         # "cut", "crossfade", or "fade_black"
  duration: 0.5              # Transition length in seconds

audio:
  music_track: null           # Path to background music (null = none)
  sync_to_beat: false
  original_audio_volume: 0.7
  music_volume: 0.3

export:
  resolution: "1920x1080"
  fps: 30
  codec: "libx264"
  bitrate: "8M"
  preset: "medium"
```

## Dependencies

- **moviepy** — Video clip loading, trimming, transitions, concatenation
- **opencv-python** — Optical flow and frame analysis for shake detection
- **scenedetect** — Content-aware scene boundary detection
- **librosa** — Audio analysis, beat detection, BPM estimation
- **click** — CLI framework
- **pyyaml** — Configuration loading
- **FFmpeg** (system) — Video probing, audio extraction, final rendering

## Design Decisions

- **Immutable Clip dataclass** — Frozen with `with_score()`, `with_metadata()`, `with_tag()` methods that return new instances. Analyzers never mutate data.
- **0–100 scoring** — All analyzers score on the same scale for intuitive comparison and filtering.
- **Config-driven** — Everything from enabled analyzers to codec settings is controlled via YAML, no code changes needed.
- **Graceful failure** — If an analyzer crashes on a clip, it's logged and skipped; the pipeline continues with the remaining analyzers.
- **Two-pass output** — Assembly creates an intermediate file, then the renderer does a final encode pass with scaling, padding, and optional music mixing.
