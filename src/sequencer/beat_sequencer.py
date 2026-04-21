"""Beat-aligned sequencer — walks the beat grid and fills slots with clips.

This module turns a pool of scored clips + a MusicMap into an EDL whose
cuts all land on beats.  It is the defining feature of the pipeline: cuts
that fall on the beat are what make auto-edits look hand-cut.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.audio.music_sync import MusicMap, MusicSection
from src.models.clip import Clip
from src.models.edl import EDL, EDLEntry

logger = logging.getLogger(__name__)


# Beat intervals per pacing label.  Range (min_beats, max_beats) between cuts.
_PACING_INTERVALS: dict[str, tuple[int, int]] = {
    "slow": (4, 8),
    "medium": (2, 4),
    "fast": (1, 2),
    "cinematic": (4, 8),
    "energetic": (1, 2),
}

# How the energy curve compresses the interval range.
_HIGH_ENERGY_THRESHOLD = 0.75
_LOW_ENERGY_THRESHOLD = 0.35


@dataclass(frozen=True, slots=True)
class BeatSlot:
    """A time window between two beats — one cut lives here.

    Attributes:
        start: Start timestamp on the final timeline (seconds).
        end: End timestamp on the final timeline (seconds).
        section_label: Label of the music section covering this slot.
        energy: Mean energy (0–1) at the slot's midpoint.
    """

    start: float
    end: float
    section_label: str
    energy: float

    @property
    def duration(self) -> float:
        """Slot length (seconds)."""
        return self.end - self.start

    @property
    def mid(self) -> float:
        """Midpoint time (seconds)."""
        return (self.start + self.end) / 2.0


def build_edl(
    clips: list[Clip],
    music_map: MusicMap,
    target_duration: float,
    pacing: str = "medium",
    mood: str = "neutral",
) -> EDL:
    """Build a beat-aligned EDL from a pool of ranked clips.

    Steps:
        1. Walk the beat grid from t=0 to *target_duration*.
        2. Group beats into slots using the pacing rules.
        3. Assign one clip per slot, preferring clips that match the
           slot's section (intro → wide, peak → best-scored, etc.).
        4. Trim each chosen clip to the slot duration.

    Args:
        clips: Ranked list of candidate clips (highest score first).
        music_map: Music analysis result.
        target_duration: Desired total edit length (seconds).
        pacing: One of ``"slow"``, ``"medium"``, ``"fast"``,
            ``"cinematic"``, ``"energetic"``.
        mood: Mood label (stored on the EDL, used downstream for colour).

    Returns:
        A populated EDL.  Empty if no viable slots can be filled.
    """
    if not clips:
        logger.warning("No clips provided to sequencer — returning empty EDL")
        return EDL(target_duration=target_duration, bpm=music_map.bpm,
                   mood=mood, pacing=pacing)

    slots = _carve_slots(music_map, target_duration, pacing)
    if not slots:
        logger.warning("No beat slots could be carved — falling back to single-shot EDL")
        return _fallback_edl(clips, target_duration, music_map, mood, pacing)

    entries = _fill_slots(slots, clips, music_map)

    logger.info(
        "Sequenced %d cuts across %.1fs  (pacing=%s, %.1f BPM, mood=%s)",
        len(entries), target_duration, pacing, music_map.bpm, mood,
    )

    return EDL(
        entries=entries,
        target_duration=target_duration,
        bpm=music_map.bpm,
        mood=mood,
        pacing=pacing,
    )


# ------------------------------------------------------------------
# Slot carving
# ------------------------------------------------------------------

def _carve_slots(
    music_map: MusicMap,
    target_duration: float,
    pacing: str,
) -> list[BeatSlot]:
    """Partition the beat grid into variable-length slots."""
    beats = [b for b in music_map.beat_times if b <= target_duration]
    if len(beats) < 2:
        return []

    interval = _PACING_INTERVALS.get(pacing, _PACING_INTERVALS["medium"])
    min_beats, max_beats = interval

    slots: list[BeatSlot] = []
    i = 0
    while i < len(beats) - 1:
        # Determine how many beats to skip based on local energy.
        t_here = beats[i]
        energy = music_map.energy_at(t_here)
        if energy >= _HIGH_ENERGY_THRESHOLD:
            step = min_beats
        elif energy <= _LOW_ENERGY_THRESHOLD:
            step = max_beats
        else:
            # Linear interpolation within the band.
            frac = (energy - _LOW_ENERGY_THRESHOLD) / (
                _HIGH_ENERGY_THRESHOLD - _LOW_ENERGY_THRESHOLD
            )
            step = int(round(max_beats - frac * (max_beats - min_beats)))
            step = max(min_beats, min(max_beats, step))

        j = min(i + step, len(beats) - 1)
        start = beats[i]
        end = beats[j]
        if end <= start:
            break

        section = music_map.section_at((start + end) / 2.0)
        slot_energy = music_map.energy_at((start + end) / 2.0)
        slots.append(BeatSlot(
            start=start,
            end=end,
            section_label=section.label if section else "sustain",
            energy=slot_energy,
        ))
        i = j

    # Cap at target_duration.
    trimmed: list[BeatSlot] = []
    for s in slots:
        if s.start >= target_duration:
            break
        if s.end > target_duration:
            s = BeatSlot(s.start, target_duration, s.section_label, s.energy)
        trimmed.append(s)
    return trimmed


# ------------------------------------------------------------------
# Slot filling
# ------------------------------------------------------------------

def _fill_slots(
    slots: list[BeatSlot],
    clips: list[Clip],
    music_map: MusicMap,
) -> list[EDLEntry]:
    """Assign one clip to each slot, trimming to fit."""
    available = list(clips)  # Mutable copy — we'll pop as we go.
    entries: list[EDLEntry] = []
    last_source: str | None = None

    for idx, slot in enumerate(slots):
        if not available:
            # Re-fill when we run out — better to repeat than leave a gap.
            available = list(clips)

        pick = _pick_clip_for_slot(
            slot=slot,
            pool=available,
            last_source=last_source,
            is_first=(idx == 0),
            is_last=(idx == len(slots) - 1),
        )
        if pick is None:
            continue

        # Compute the best sub-range within the source clip.
        src_in, src_out = _choose_source_window(pick, slot.duration)

        # Mark intro/peak/outro.
        notes = slot.section_label
        if music_map.peak_time and abs(slot.mid - music_map.peak_time) < 1.5:
            notes = "peak"

        entries.append(EDLEntry(
            source_path=pick.path,
            source_in=src_in,
            source_out=src_out,
            timeline_in=slot.start,
            duration=slot.duration,
            transition_in="cut",
            transition_out="cut",
            beat_aligned=True,
            shot_type=pick.metadata.get("shot_type") if pick.metadata else None,
            movement=pick.metadata.get("movement") if pick.metadata else None,
            score=pick.composite_score,
            notes=notes,
        ))

        last_source = str(pick.path)
        # Don't immediately reuse — move to the back of the queue.
        try:
            available.remove(pick)
        except ValueError:
            pass

    return entries


def _pick_clip_for_slot(
    slot: BeatSlot,
    pool: list[Clip],
    last_source: str | None,
    is_first: bool,
    is_last: bool,
) -> Clip | None:
    """Pick the best clip for *slot* from *pool*, avoiding back-to-back repeats."""
    if not pool:
        return None

    def _score_for(clip: Clip) -> float:
        s = clip.composite_score
        # Prefer length >= slot duration so we don't time-stretch.
        if clip.duration >= slot.duration:
            s += 10.0
        # Penalise immediate repeats.
        if last_source and str(clip.path) == last_source:
            s -= 30.0
        # Section-specific bonuses using shot_type metadata if present.
        shot_type = clip.metadata.get("shot_type") if clip.metadata else None
        if is_first and shot_type == "wide_establishing":
            s += 15.0
        if slot.section_label == "peak" and shot_type in {"reveal", "push_in"}:
            s += 20.0
        if is_last and shot_type == "pull_out":
            s += 15.0
        # Energy-appropriate movement.
        movement = clip.metadata.get("movement") if clip.metadata else None
        if slot.energy >= _HIGH_ENERGY_THRESHOLD and movement in {"push_in", "tracking", "orbit"}:
            s += 5.0
        if slot.energy <= _LOW_ENERGY_THRESHOLD and movement in {"static", "reveal"}:
            s += 5.0
        return s

    return max(pool, key=_score_for)


def _choose_source_window(clip: Clip, needed: float) -> tuple[float, float]:
    """Pick the best sub-range of the source clip for a slot of *needed* seconds."""
    if clip.duration <= needed:
        return (0.0, clip.duration)
    # Prefer the middle — skip first/last 10 % which are often ramp-in/out.
    pad = clip.duration * 0.1
    usable_start = pad
    usable_end = clip.duration - pad
    usable_dur = usable_end - usable_start
    if usable_dur >= needed:
        mid = (usable_start + usable_end) / 2.0
        return (mid - needed / 2.0, mid + needed / 2.0)
    return (0.0, needed)


# ------------------------------------------------------------------
# Fallback when no beat grid is usable
# ------------------------------------------------------------------

def _fallback_edl(
    clips: list[Clip],
    target_duration: float,
    music_map: MusicMap,
    mood: str,
    pacing: str,
) -> EDL:
    """Evenly-spaced fallback when we can't carve beat slots."""
    if not clips:
        return EDL(target_duration=target_duration, bpm=music_map.bpm,
                   mood=mood, pacing=pacing)

    n = min(len(clips), max(1, int(target_duration / 3.0)))
    dur_each = target_duration / n
    entries: list[EDLEntry] = []
    for i, c in enumerate(clips[:n]):
        src_in, src_out = _choose_source_window(c, dur_each)
        entries.append(EDLEntry(
            source_path=c.path,
            source_in=src_in,
            source_out=src_out,
            timeline_in=i * dur_each,
            duration=dur_each,
            beat_aligned=False,
            score=c.composite_score,
            notes="fallback",
        ))
    return EDL(entries=entries, target_duration=target_duration,
               bpm=music_map.bpm, mood=mood, pacing=pacing)
