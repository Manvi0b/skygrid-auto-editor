"""Source profile — describes the device that captured the footage."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class SourceProfile:
    """Describes a camera/device source and its known characteristics.

    Used to inform analyzers about expected artifacts and to help the
    assembler make orientation decisions.

    Attributes:
        name: Machine-readable identifier (e.g. ``"dji_mini3pro"``).
        device_type: Category of capture device — ``"drone"``, ``"gimbal"``,
            ``"handheld"``, or ``"generic"``.
        default_orientation: Expected frame orientation — ``"horizontal"``,
            ``"vertical"``, or ``"mixed"``.
        has_gimbal: Whether the device has a stabilisation gimbal.
        typical_artifacts: Known artifacts to watch for during analysis
            (e.g. ``["prop_shadow", "jello"]`` for consumer drones).
    """

    name: str
    device_type: str = "generic"
    default_orientation: str = "horizontal"
    has_gimbal: bool = False
    typical_artifacts: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Built-in profiles
# ---------------------------------------------------------------------------

BUILTIN_SOURCE_PROFILES: dict[str, SourceProfile] = {
    "dji_mini3pro": SourceProfile(
        name="dji_mini3pro",
        device_type="drone",
        default_orientation="horizontal",
        has_gimbal=True,
        typical_artifacts=["prop_shadow", "jello"],
    ),
    "dji_mini4pro": SourceProfile(
        name="dji_mini4pro",
        device_type="drone",
        default_orientation="horizontal",
        has_gimbal=True,
        typical_artifacts=["prop_shadow", "jello"],
    ),
    "dji_air3": SourceProfile(
        name="dji_air3",
        device_type="drone",
        default_orientation="horizontal",
        has_gimbal=True,
        typical_artifacts=["prop_shadow"],
    ),
    "dji_mavic3": SourceProfile(
        name="dji_mavic3",
        device_type="drone",
        default_orientation="horizontal",
        has_gimbal=True,
        typical_artifacts=["prop_shadow"],
    ),
    "osmo_pocket3": SourceProfile(
        name="osmo_pocket3",
        device_type="gimbal",
        default_orientation="mixed",
        has_gimbal=True,
        typical_artifacts=[],
    ),
    "osmo_action5": SourceProfile(
        name="osmo_action5",
        device_type="handheld",
        default_orientation="horizontal",
        has_gimbal=False,
        typical_artifacts=["rolling_shutter"],
    ),
    "gopro_hero12": SourceProfile(
        name="gopro_hero12",
        device_type="handheld",
        default_orientation="horizontal",
        has_gimbal=False,
        typical_artifacts=["fisheye", "rolling_shutter"],
    ),
    "generic": SourceProfile(
        name="generic",
        device_type="generic",
        default_orientation="horizontal",
        has_gimbal=False,
        typical_artifacts=[],
    ),
}
