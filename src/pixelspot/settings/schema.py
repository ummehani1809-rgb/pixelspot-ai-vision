"""Configuration schema.

This module is the single authoritative description of every PixelSpot setting:
its type, its legal range and how it relates to the rest of the config. It is a
plain pydantic model tree with no I/O, so it can be validated from a dict in a
test without touching the filesystem. Loading and layering live in
``settings.loader``.

Two rules make the schema catch mistakes rather than absorb them:

* every model sets ``extra="forbid"``, so a misspelled key is an error instead
  of a silently ignored one;
* :meth:`PixelSpotConfig.validate_cross_references` reports *all* problems at
  once rather than aborting on the first, because fixing a config one error per
  run is miserable.

Analytics blocks for features that are not implemented yet are defined here on
purpose. Declaring the contract up front is what lets a new processor be added
in later phases without reshaping the config file.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Attribute backends that actually exist. A config asking for anything else is
# a configuration error rather than a feature that silently never runs.
KNOWN_ATTRIBUTE_BACKENDS = frozenset({"none", "opencv_dnn", "insightface"})

_DURATION_RE = re.compile(r"^(\d+)(s|m|h|d)$")
_DURATION_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400}

_DEVICE_RE = re.compile(r"^(auto|cpu|mps|cuda(:\d+)?)$")


def parse_duration(value: str) -> float:
    """Convert a duration such as ``15m`` into seconds."""
    match = _DURATION_RE.match(value)
    if match is None:
        raise ValueError(
            f"invalid duration {value!r}: expected <number><s|m|h|d>, e.g. '15m'"
        )
    amount, unit = match.groups()
    return int(amount) * _DURATION_UNITS[unit]


class _Base(BaseModel):
    """Shared behaviour for every config model."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
    )


# ==================================================================
# RUNTIME
# ==================================================================


class RuntimeConfig(_Base):
    device: str = "auto"
    headless: bool = False
    target_fps: float = Field(15.0, gt=0, le=240)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_file: str | None = None
    show_fps: bool = True
    window_name: str = "PixelSpot AI Analytics"

    @field_validator("device")
    @classmethod
    def _check_device(cls, value: str) -> str:
        normalised = value.lower()
        if not _DEVICE_RE.match(normalised):
            raise ValueError(
                f"invalid device {value!r}: expected 'auto', 'cpu', 'mps', "
                "'cuda' or 'cuda:<index>'"
            )
        return normalised


# ==================================================================
# SOURCE
# ==================================================================


class ReconnectConfig(_Base):
    enabled: bool = True
    initial_backoff_s: float = Field(1.0, gt=0)
    max_backoff_s: float = Field(30.0, gt=0)
    max_attempts: int = Field(0, ge=0, description="0 means retry forever")

    @model_validator(mode="after")
    def _check_backoff(self) -> "ReconnectConfig":
        if self.initial_backoff_s > self.max_backoff_s:
            raise ValueError(
                "source.reconnect.initial_backoff_s must not exceed max_backoff_s"
            )
        return self


class BufferConfig(_Base):
    max_queue: int = Field(4, ge=1, le=256)
    drop_policy: Literal["drop_oldest", "drop_newest", "block"] = "drop_oldest"


class SourceConfig(_Base):
    type: Literal["file", "webcam", "rtsp", "http"] = "file"
    uri: str
    loop: bool = False
    # File sources only: play the recording at its own speed, skipping frames
    # inference is too slow to reach -- as a live camera would have. Off means
    # every frame is analysed even if playback runs slower than real time.
    realtime: bool = False
    reconnect: ReconnectConfig = Field(default_factory=ReconnectConfig)
    buffer: BufferConfig = Field(default_factory=BufferConfig)

    @model_validator(mode="after")
    def _check_uri(self) -> "SourceConfig":
        if self.type == "rtsp" and not self.uri.lower().startswith("rtsp://"):
            raise ValueError("source.uri must start with rtsp:// when type is 'rtsp'")
        if self.type == "http" and not self.uri.lower().startswith(("http://", "https://")):
            raise ValueError("source.uri must start with http(s):// when type is 'http'")
        if self.type == "webcam" and not self.uri.isdigit():
            raise ValueError(
                "source.uri must be a camera index such as '0' when type is 'webcam'"
            )
        if self.type == "file" and self.uri.lower().startswith(("rtsp://", "http://")):
            raise ValueError("source.type is 'file' but source.uri looks like a stream URL")
        return self


# ==================================================================
# PERCEPTION
# ==================================================================


class DetectorConfig(_Base):
    model: str = "models/yolo11n.pt"
    imgsz: int = Field(640, ge=32, le=4096)
    conf: float = Field(0.35, ge=0.0, le=1.0)
    iou: float = Field(0.5, ge=0.0, le=1.0)
    max_det: int = Field(300, ge=1)
    half: bool = False
    classes: list[str] = Field(default_factory=lambda: ["person"])

    @field_validator("imgsz")
    @classmethod
    def _check_stride(cls, value: int) -> int:
        if value % 32 != 0:
            raise ValueError(
                f"perception.detector.imgsz must be a multiple of 32, got {value}"
            )
        return value

    @field_validator("classes")
    @classmethod
    def _check_classes(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("perception.detector.classes must not be empty")
        return value


class TrackerConfig(_Base):
    type: Literal["bytetrack", "botsort"] = "bytetrack"
    config: str = "bytetrack.yaml"
    max_age_s: float = Field(2.0, gt=0)
    min_hits: int = Field(3, ge=1)


class HeadPoseConfig(_Base):
    enabled: bool = False
    backend: Literal["pose_keypoints", "face_mesh"] = "pose_keypoints"
    model: str = "models/yolo11n-pose.pt"
    every_n_frames: int = Field(2, ge=1)
    min_keypoint_confidence: float = Field(0.4, ge=0.0, le=1.0)


class FaceConfig(_Base):
    enabled: bool = False
    model: str | None = None
    # Below this the classifiers upscale mush and guess; 48px is roughly the
    # smallest face the 96px attribute models still read reliably.
    min_size_px: int = Field(48, ge=1)
    every_n_frames: int = Field(5, ge=1)


class EnrichmentConfig(_Base):
    head_pose: HeadPoseConfig = Field(default_factory=HeadPoseConfig)
    face: FaceConfig = Field(default_factory=FaceConfig)


class PerceptionConfig(_Base):
    detector: DetectorConfig = Field(default_factory=DetectorConfig)
    tracker: TrackerConfig = Field(default_factory=TrackerConfig)
    enrichment: EnrichmentConfig = Field(default_factory=EnrichmentConfig)


# ==================================================================
# GEOMETRY
# ==================================================================

Point = tuple[float, float]


class ZoneConfig(_Base):
    id: str
    points: list[Point] = Field(min_length=3)
    tags: list[str] = Field(default_factory=list)


class LineConfig(_Base):
    id: str
    p1: Point
    p2: Point
    positive: Literal["down", "up", "left", "right"] = "down"
    classes: list[str] = Field(default_factory=lambda: ["person"])

    @model_validator(mode="after")
    def _check_distinct(self) -> "LineConfig":
        if self.p1 == self.p2:
            raise ValueError(f"line {self.id!r}: p1 and p2 must be different points")
        return self


class ScreenConfig(_Base):
    id: str
    position: Point
    facing: Point = (0.0, 1.0)
    fov_deg: float = Field(120.0, gt=0, le=360)
    max_distance_m: float | None = Field(None, gt=0)

    @model_validator(mode="after")
    def _check_facing(self) -> "ScreenConfig":
        if self.facing == (0.0, 0.0):
            raise ValueError(f"screen {self.id!r}: facing must be a non-zero vector")
        return self


class GeometryConfig(_Base):
    coordinate_space: Literal["normalized", "pixels"] = "normalized"
    zones: list[ZoneConfig] = Field(default_factory=list)
    lines: list[LineConfig] = Field(default_factory=list)
    screens: list[ScreenConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_geometry(self) -> "GeometryConfig":
        errors: list[str] = []

        for label, items in (
            ("zone", self.zones),
            ("line", self.lines),
            ("screen", self.screens),
        ):
            seen: set[str] = set()
            for item in items:
                if item.id in seen:
                    errors.append(f"geometry: duplicate {label} id {item.id!r}")
                seen.add(item.id)

        if self.coordinate_space == "normalized":
            def out_of_range(point: Point) -> bool:
                return not (0.0 <= point[0] <= 1.0 and 0.0 <= point[1] <= 1.0)

            for zone in self.zones:
                for point in zone.points:
                    if out_of_range(point):
                        errors.append(
                            f"geometry.zones[{zone.id!r}]: point {point} is outside "
                            "0.0-1.0 but coordinate_space is 'normalized'"
                        )
            for line in self.lines:
                for point in (line.p1, line.p2):
                    if out_of_range(point):
                        errors.append(
                            f"geometry.lines[{line.id!r}]: point {point} is outside "
                            "0.0-1.0 but coordinate_space is 'normalized'"
                        )
            for screen in self.screens:
                if out_of_range(screen.position):
                    errors.append(
                        f"geometry.screens[{screen.id!r}]: position "
                        f"{screen.position} is outside 0.0-1.0"
                    )

        if errors:
            raise ValueError("\n".join(errors))
        return self

    def zone_ids(self) -> set[str]:
        return {zone.id for zone in self.zones}

    def line_ids(self) -> set[str]:
        return {line.id for line in self.lines}

    def screen_ids(self) -> set[str]:
        return {screen.id for screen in self.screens}


# ==================================================================
# ANALYTICS
# ==================================================================


class ProcessorConfig(_Base):
    """Base for every analytics processor block."""

    enabled: bool = False


class _AttributeProcessorConfig(ProcessorConfig):
    """Shared shape for the per-track attribute classifiers.

    These are sampled every N frames and voted over a window rather than being
    recomputed per frame: running a classifier on every face on every frame is
    both slower and less stable than voting on a track.
    """

    backend: str = "none"
    model: str | None = None
    every_n_frames: int = Field(5, ge=1)
    vote_window: int = Field(15, ge=1)
    min_confidence: float = Field(0.6, ge=0.0, le=1.0)

    @field_validator("backend")
    @classmethod
    def _check_backend(cls, value: str) -> str:
        if value not in KNOWN_ATTRIBUTE_BACKENDS:
            known = ", ".join(sorted(KNOWN_ATTRIBUTE_BACKENDS))
            raise ValueError(f"unknown backend {value!r}; available: {known}")
        return value


class FootfallConfig(ProcessorConfig):
    lines: list[str] = Field(default_factory=list)
    classes: list[str] = Field(default_factory=lambda: ["person"])
    hysteresis_px: int = Field(12, ge=0)
    min_track_age_frames: int = Field(3, ge=0)
    cooldown_s: float = Field(1.0, ge=0)


class ViewingZoneConfig(ProcessorConfig):
    zones: list[str] = Field(default_factory=list)
    classes: list[str] = Field(default_factory=lambda: ["person"])


class DwellConfig(ProcessorConfig):
    zones: list[str] = Field(default_factory=list)
    min_dwell_s: float = Field(2.0, ge=0)
    max_dwell_s: float = Field(3600.0, gt=0)

    @model_validator(mode="after")
    def _check_bounds(self) -> "DwellConfig":
        if self.min_dwell_s >= self.max_dwell_s:
            raise ValueError("analytics.dwell.min_dwell_s must be below max_dwell_s")
        return self


class AttentionConfig(ProcessorConfig):
    zones: list[str] = Field(default_factory=list)
    screen: str | None = None
    yaw_tolerance_deg: float = Field(35.0, gt=0, le=180)
    min_gaze_s: float = Field(1.0, ge=0)


class ScreenVisibilityConfig(ProcessorConfig):
    screen: str | None = None
    max_distance_m: float | None = Field(None, gt=0)


class DensityLevels(_Base):
    low: float = Field(0.3, ge=0)
    medium: float = Field(0.7, ge=0)
    high: float = Field(1.2, ge=0)

    @model_validator(mode="after")
    def _check_ascending(self) -> "DensityLevels":
        if not self.low < self.medium < self.high:
            raise ValueError(
                "analytics.crowd_density.levels must be ascending: low < medium < high"
            )
        return self


class CrowdDensityConfig(ProcessorConfig):
    zones: list[str] = Field(default_factory=list)
    zone_areas_m2: dict[str, float] = Field(default_factory=dict)
    levels: DensityLevels = Field(default_factory=DensityLevels)

    @field_validator("zone_areas_m2")
    @classmethod
    def _check_areas(cls, value: dict[str, float]) -> dict[str, float]:
        for zone_id, area in value.items():
            if area <= 0:
                raise ValueError(
                    f"analytics.crowd_density.zone_areas_m2[{zone_id!r}] must be > 0"
                )
        return value


class QueueConfig(ProcessorConfig):
    zones: list[str] = Field(default_factory=list)
    min_people: int = Field(3, ge=1)
    min_dwell_s: float = Field(10.0, ge=0)
    cluster_radius_px: int = Field(120, gt=0)


class TrafficDirectionConfig(ProcessorConfig):
    smoothing_frames: int = Field(8, ge=1)
    bins: int = Field(8, ge=2, le=36)
    min_speed_px_s: float = Field(15.0, ge=0)


class AudienceFlowConfig(ProcessorConfig):
    zones: list[str] = Field(default_factory=list)
    min_transition_s: float = Field(0.5, ge=0)


class HeatmapConfig(ProcessorConfig):
    grid: tuple[int, int] = (64, 36)
    decay: float = Field(0.98, gt=0, le=1.0)
    weight_by: Literal["presence", "attention", "dwell"] = "presence"

    @field_validator("grid")
    @classmethod
    def _check_grid(cls, value: tuple[int, int]) -> tuple[int, int]:
        if value[0] < 2 or value[1] < 2:
            raise ValueError("analytics.heatmap.grid cells must be at least 2x2")
        return value


class GenderConfig(_AttributeProcessorConfig):
    labels: list[str] = Field(default_factory=lambda: ["male", "female"])


class AgeConfig(_AttributeProcessorConfig):
    buckets: list[tuple[int, int]] = Field(
        default_factory=lambda: [(0, 12), (13, 19), (20, 34), (35, 54), (55, 120)]
    )

    @field_validator("buckets")
    @classmethod
    def _check_buckets(cls, value: list[tuple[int, int]]) -> list[tuple[int, int]]:
        if not value:
            raise ValueError("analytics.age.buckets must not be empty")
        previous_high = -1
        for low, high in value:
            if low > high:
                raise ValueError(f"analytics.age.buckets: ({low}, {high}) is inverted")
            if low <= previous_high:
                raise ValueError(
                    f"analytics.age.buckets: ({low}, {high}) overlaps the previous bucket"
                )
            previous_high = high
        return value


class MoodConfig(_AttributeProcessorConfig):
    labels: list[str] = Field(
        default_factory=lambda: ["neutral", "happy", "sad", "angry", "surprised"]
    )


class VehicleConfig(ProcessorConfig):
    classes: list[str] = Field(
        default_factory=lambda: ["car", "motorcycle", "bus", "truck"]
    )
    classify: bool = True
    zones: list[str] = Field(default_factory=list)
    lines: list[str] = Field(default_factory=list)


class ParkingConfig(ProcessorConfig):
    bays: list[str] = Field(default_factory=list)
    occupied_after_s: float = Field(30.0, ge=0)
    vacated_after_s: float = Field(15.0, ge=0)


class AnomalyRule(_Base):
    id: str
    metric: str
    condition: Literal["above", "below", "spike", "drop"]
    threshold: float
    window: str = "5m"

    @field_validator("window")
    @classmethod
    def _check_window(cls, value: str) -> str:
        parse_duration(value)
        return value


class AnomalyConfig(ProcessorConfig):
    rules: list[AnomalyRule] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_rule_ids(self) -> "AnomalyConfig":
        seen: set[str] = set()
        for rule in self.rules:
            if rule.id in seen:
                raise ValueError(f"analytics.anomaly: duplicate rule id {rule.id!r}")
            seen.add(rule.id)
        return self


class AnalyticsConfig(_Base):
    footfall: FootfallConfig = Field(default_factory=FootfallConfig)
    viewing_zone: ViewingZoneConfig = Field(default_factory=ViewingZoneConfig)
    dwell: DwellConfig = Field(default_factory=DwellConfig)
    attention: AttentionConfig = Field(default_factory=AttentionConfig)
    screen_visibility: ScreenVisibilityConfig = Field(
        default_factory=ScreenVisibilityConfig
    )
    crowd_density: CrowdDensityConfig = Field(default_factory=CrowdDensityConfig)
    queue: QueueConfig = Field(default_factory=QueueConfig)
    traffic_direction: TrafficDirectionConfig = Field(
        default_factory=TrafficDirectionConfig
    )
    audience_flow: AudienceFlowConfig = Field(default_factory=AudienceFlowConfig)
    heatmap: HeatmapConfig = Field(default_factory=HeatmapConfig)
    gender: GenderConfig = Field(default_factory=GenderConfig)
    age: AgeConfig = Field(default_factory=AgeConfig)
    mood: MoodConfig = Field(default_factory=MoodConfig)
    vehicles: VehicleConfig = Field(default_factory=VehicleConfig)
    parking: ParkingConfig = Field(default_factory=ParkingConfig)
    anomaly: AnomalyConfig = Field(default_factory=AnomalyConfig)

    def items(self) -> list[tuple[str, ProcessorConfig]]:
        """Every processor block as ``(name, config)`` in declaration order."""
        return [(name, getattr(self, name)) for name in type(self).model_fields]

    def enabled_names(self) -> list[str]:
        return [name for name, block in self.items() if block.enabled]


# ==================================================================
# AGGREGATION AND SINKS
# ==================================================================


class AggregationConfig(_Base):
    windows: list[str] = Field(default_factory=lambda: ["1m", "15m", "1h"])
    emit_interval_s: float = Field(60.0, gt=0)
    dedupe_events: bool = True

    @field_validator("windows")
    @classmethod
    def _check_windows(cls, value: list[str]) -> list[str]:
        for window in value:
            parse_duration(window)
        return value

    def window_seconds(self) -> dict[str, float]:
        return {window: parse_duration(window) for window in self.windows}


class ConsoleSinkConfig(_Base):
    enabled: bool = True
    emit: Literal["events", "metrics", "both"] = "events"


class JsonlSinkConfig(_Base):
    enabled: bool = False
    path: str = "out/metrics.jsonl"
    rotate_mb: int = Field(64, ge=1)


class RetryConfig(_Base):
    max_attempts: int = Field(5, ge=0)
    backoff_s: float = Field(2.0, gt=0)


class CCMSSinkConfig(_Base):
    enabled: bool = False
    base_url: str | None = None
    api_key: str | None = None
    device_id: str | None = None
    batch_size: int = Field(50, ge=1)
    timeout_s: float = Field(10.0, gt=0)
    retry: RetryConfig = Field(default_factory=RetryConfig)
    spool_dir: str = "out/spool"

    @model_validator(mode="after")
    def _check_required_when_enabled(self) -> "CCMSSinkConfig":
        if not self.enabled:
            return self
        missing = [
            name
            for name in ("base_url", "api_key", "device_id")
            if not getattr(self, name)
        ]
        if missing:
            raise ValueError(
                "sinks.ccms is enabled but missing: " + ", ".join(sorted(missing))
            )
        return self


class SinksConfig(_Base):
    console: ConsoleSinkConfig = Field(default_factory=ConsoleSinkConfig)
    jsonl: JsonlSinkConfig = Field(default_factory=JsonlSinkConfig)
    ccms: CCMSSinkConfig = Field(default_factory=CCMSSinkConfig)


class PrivacyConfig(_Base):
    """Defaults are deliberately the restrictive ones.

    Age, gender and mood inference on faces is biometric processing under GDPR
    and the DPDP Act. Opting *in* to retaining imagery should be a deliberate
    edit, not something a fresh install does by accident.
    """

    store_crops: bool = False
    blur_faces_in_output: bool = True
    aggregate_only: bool = True


# ==================================================================
# ROOT
# ==================================================================


class PixelSpotConfig(_Base):
    version: int = Field(1, ge=1)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    source: SourceConfig
    perception: PerceptionConfig = Field(default_factory=PerceptionConfig)
    geometry: GeometryConfig = Field(default_factory=GeometryConfig)
    analytics: AnalyticsConfig = Field(default_factory=AnalyticsConfig)
    aggregation: AggregationConfig = Field(default_factory=AggregationConfig)
    sinks: SinksConfig = Field(default_factory=SinksConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)

    @model_validator(mode="after")
    def validate_cross_references(self) -> "PixelSpotConfig":
        """Check settings that only make sense relative to other settings.

        Collects every problem before raising so one run reports the whole list.
        """
        errors: list[str] = []
        analytics = self.analytics
        zones = self.geometry.zone_ids()
        lines = self.geometry.line_ids()
        screens = self.geometry.screen_ids()

        def check_refs(
            block: ProcessorConfig,
            name: str,
            field: str,
            refs: list[str],
            known: set[str],
            kind: str,
        ) -> None:
            if not block.enabled:
                return
            for ref in refs:
                if ref not in known:
                    available = ", ".join(sorted(known)) or "none defined"
                    errors.append(
                        f"analytics.{name}.{field}: unknown {kind} {ref!r} "
                        f"(available: {available})"
                    )

        check_refs(analytics.footfall, "footfall", "lines",
                   analytics.footfall.lines, lines, "line")
        check_refs(analytics.viewing_zone, "viewing_zone", "zones",
                   analytics.viewing_zone.zones, zones, "zone")
        check_refs(analytics.dwell, "dwell", "zones",
                   analytics.dwell.zones, zones, "zone")
        check_refs(analytics.attention, "attention", "zones",
                   analytics.attention.zones, zones, "zone")
        check_refs(analytics.crowd_density, "crowd_density", "zones",
                   analytics.crowd_density.zones, zones, "zone")
        check_refs(analytics.queue, "queue", "zones",
                   analytics.queue.zones, zones, "zone")
        check_refs(analytics.audience_flow, "audience_flow", "zones",
                   analytics.audience_flow.zones, zones, "zone")
        check_refs(analytics.vehicles, "vehicles", "zones",
                   analytics.vehicles.zones, zones, "zone")
        check_refs(analytics.vehicles, "vehicles", "lines",
                   analytics.vehicles.lines, lines, "line")
        check_refs(analytics.parking, "parking", "bays",
                   analytics.parking.bays, zones, "zone")

        if analytics.attention.enabled and analytics.attention.screen:
            check_refs(analytics.attention, "attention", "screen",
                       [analytics.attention.screen], screens, "screen")
        if analytics.screen_visibility.enabled and analytics.screen_visibility.screen:
            check_refs(analytics.screen_visibility, "screen_visibility", "screen",
                       [analytics.screen_visibility.screen], screens, "screen")

        # Processors that need a perception capability switched on.
        head_pose_on = self.perception.enrichment.head_pose.enabled
        face_on = self.perception.enrichment.face.enabled

        if analytics.attention.enabled and not head_pose_on:
            errors.append(
                "analytics.attention is enabled but "
                "perception.enrichment.head_pose.enabled is false"
            )
        if analytics.screen_visibility.enabled and not head_pose_on:
            errors.append(
                "analytics.screen_visibility is enabled but "
                "perception.enrichment.head_pose.enabled is false"
            )
        for name in ("gender", "age", "mood"):
            block = getattr(analytics, name)
            if block.enabled and block.backend != "none" and not face_on:
                errors.append(
                    f"analytics.{name} uses backend {block.backend!r} but "
                    "perception.enrichment.face.enabled is false"
                )

        # Processors that are meaningless without their own targets.
        if analytics.footfall.enabled and not analytics.footfall.lines:
            errors.append("analytics.footfall is enabled but lists no lines")
        if analytics.viewing_zone.enabled and not analytics.viewing_zone.zones:
            errors.append("analytics.viewing_zone is enabled but lists no zones")
        if analytics.parking.enabled and not analytics.parking.bays:
            errors.append("analytics.parking is enabled but lists no bays")
        if analytics.anomaly.enabled and not analytics.anomaly.rules:
            errors.append("analytics.anomaly is enabled but defines no rules")

        if analytics.crowd_density.enabled:
            for zone_id in analytics.crowd_density.zones:
                if zone_id not in analytics.crowd_density.zone_areas_m2:
                    errors.append(
                        f"analytics.crowd_density.zone_areas_m2 has no entry for "
                        f"zone {zone_id!r}; density cannot be computed without it"
                    )

        # Detector must be told to look for the classes the processors want.
        detector_classes = set(self.perception.detector.classes)
        if analytics.footfall.enabled:
            missing = set(analytics.footfall.classes) - detector_classes
            if missing:
                errors.append(
                    "analytics.footfall.classes contains "
                    f"{sorted(missing)} which perception.detector.classes does not detect"
                )
        if analytics.vehicles.enabled:
            missing = set(analytics.vehicles.classes) - detector_classes
            if missing:
                errors.append(
                    "analytics.vehicles.classes contains "
                    f"{sorted(missing)} which perception.detector.classes does not detect"
                )

        # Privacy guard rails.
        if self.privacy.aggregate_only and self.privacy.store_crops:
            errors.append(
                "privacy.aggregate_only and privacy.store_crops cannot both be true"
            )

        if errors:
            raise ValueError("\n".join(errors))
        return self

    def enabled_features(self) -> list[str]:
        return self.analytics.enabled_names()
