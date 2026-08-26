"""Tests for the configuration layer.

These run in milliseconds: no video, no model, no GPU. Every check that the
loader is supposed to make is pinned here, so a future change that quietly
stops rejecting a bad config fails the suite instead of reaching a device.
"""

from __future__ import annotations

import textwrap

import pytest
from pydantic import ValidationError

from pixelspot.settings.loader import (
    ConfigError,
    deep_merge,
    env_overrides,
    expand_variables,
    load_config,
)
from pixelspot.settings.schema import PixelSpotConfig, parse_duration

# Source, perception and geometry, with no analytics block. Tests append the
# analytics they care about so no test ends up with two `analytics:` keys.
BASE = """
source:
  type: file
  uri: datasets/crowd.mp4
perception:
  detector:
    classes: [person, car]
geometry:
  zones:
    - id: storefront
      points: [[0.1, 0.1], [0.3, 0.1], [0.3, 0.5], [0.1, 0.5]]
  lines:
    - id: entrance
      p1: [0.0, 0.37]
      p2: [1.0, 0.37]
"""

FOOTFALL = """
analytics:
  footfall:
    enabled: true
    lines: [entrance]
"""


def write_config(tmp_path, *blocks: str, name: str = "config.yaml"):
    """Write the YAML formed by joining *blocks*, each dedented on its own.

    Dedenting per block matters: joining first and dedenting after leaves the
    indentation of later blocks intact and produces invalid YAML.
    """
    text = "\n".join(textwrap.dedent(block).strip("\n") for block in blocks)
    path = tmp_path / name
    path.write_text(text + "\n", encoding="utf-8")
    return path


# ==================================================================
# HAPPY PATH
# ==================================================================


def test_minimal_config_loads(tmp_path):
    config, report = load_config(write_config(tmp_path, BASE, FOOTFALL), environ={})

    assert config.source.uri == "datasets/crowd.mp4"
    assert config.analytics.footfall.enabled is True
    assert config.enabled_features() == ["footfall"]
    assert report.env_keys == []


def test_defaults_fill_in_omitted_sections(tmp_path):
    config, _ = load_config(write_config(tmp_path, BASE, FOOTFALL), environ={})

    assert config.runtime.device == "auto"
    assert config.runtime.log_level == "INFO"
    assert config.perception.tracker.type == "bytetrack"
    assert config.privacy.aggregate_only is True
    assert config.analytics.mood.enabled is False


def test_every_processor_block_exists():
    """The schema must already accommodate all planned capabilities."""
    expected = {
        "footfall", "viewing_zone", "dwell", "attention", "screen_visibility",
        "crowd_density", "queue", "traffic_direction", "audience_flow",
        "heatmap", "gender", "age", "mood", "vehicles", "parking", "anomaly",
    }
    names = {name for name, _ in PixelSpotConfig.model_fields["analytics"]
             .annotation.model_fields.items()}
    assert expected <= names


# ==================================================================
# STRUCTURAL VALIDATION
# ==================================================================


def test_unknown_key_is_rejected(tmp_path):
    path = write_config(tmp_path, BASE, FOOTFALL, "runtime:\n  headles: true")
    with pytest.raises(ConfigError, match="headles"):
        load_config(path, environ={})


def test_unknown_top_level_section_is_rejected(tmp_path):
    path = write_config(tmp_path, BASE, "analitics:\n  footfall:\n    enabled: true")
    with pytest.raises(ConfigError, match="analitics"):
        load_config(path, environ={})


@pytest.mark.parametrize(
    "override, expected",
    [
        ("perception:\n  detector:\n    conf: 1.7\n", "conf"),
        ("perception:\n  detector:\n    imgsz: 500\n", "multiple of 32"),
        ("runtime:\n  device: gpu0\n", "device"),
        ("runtime:\n  target_fps: 0\n", "target_fps"),
        ("aggregation:\n  windows: [15x]\n", "duration"),
    ],
)
def test_out_of_range_values_are_rejected(tmp_path, override, expected):
    with pytest.raises(ConfigError, match=expected):
        load_config(write_config(tmp_path, BASE, FOOTFALL, override), environ={})


def test_normalized_coordinates_must_be_within_unit_range(tmp_path):
    path = write_config(tmp_path, """
        source: {type: file, uri: a.mp4}
        geometry:
          zones:
            - id: bad
              points: [[0.1, 0.1], [1.4, 0.1], [0.3, 0.5]]
    """)
    with pytest.raises(ConfigError, match="outside 0.0-1.0"):
        load_config(path, environ={})


def test_rtsp_type_requires_rtsp_uri(tmp_path):
    path = write_config(tmp_path, "source: {type: rtsp, uri: datasets/crowd.mp4}\n")
    with pytest.raises(ConfigError, match="rtsp://"):
        load_config(path, environ={})


def test_duplicate_zone_ids_are_rejected(tmp_path):
    path = write_config(tmp_path, """
        source: {type: file, uri: a.mp4}
        geometry:
          zones:
            - {id: dup, points: [[0.1,0.1],[0.2,0.1],[0.2,0.2]]}
            - {id: dup, points: [[0.3,0.3],[0.4,0.3],[0.4,0.4]]}
    """)
    with pytest.raises(ConfigError, match="duplicate zone id"):
        load_config(path, environ={})


# ==================================================================
# CROSS-REFERENCE VALIDATION
# ==================================================================


def test_processor_referencing_unknown_line_is_rejected(tmp_path):
    path = write_config(
        tmp_path, BASE, FOOTFALL.replace("lines: [entrance]", "lines: [front_door]")
    )
    with pytest.raises(ConfigError, match="unknown line 'front_door'"):
        load_config(path, environ={})


def test_attention_requires_head_pose(tmp_path):
    path = write_config(tmp_path, BASE, """
        analytics:
          footfall: {enabled: true, lines: [entrance]}
          attention: {enabled: true, zones: [storefront]}
    """)
    with pytest.raises(ConfigError, match="head_pose"):
        load_config(path, environ={})


def test_enabled_processor_without_targets_is_rejected(tmp_path):
    path = write_config(tmp_path, BASE, """
        analytics:
          footfall: {enabled: true, lines: [entrance]}
          viewing_zone: {enabled: true}
    """)
    with pytest.raises(ConfigError, match="lists no zones"):
        load_config(path, environ={})


def test_processor_class_must_be_detected(tmp_path):
    """Enabling vehicles while the detector ignores vehicles reports zero forever."""
    path = write_config(tmp_path, BASE, """
        analytics:
          footfall: {enabled: true, lines: [entrance]}
          vehicles: {enabled: true, classes: [bus, truck]}
    """)
    with pytest.raises(ConfigError, match="does not detect"):
        load_config(path, environ={})


def test_crowd_density_requires_zone_area(tmp_path):
    path = write_config(tmp_path, BASE, """
        analytics:
          footfall: {enabled: true, lines: [entrance]}
          crowd_density: {enabled: true, zones: [storefront]}
    """)
    with pytest.raises(ConfigError, match="zone_areas_m2"):
        load_config(path, environ={})


def test_unimplemented_backend_is_rejected(tmp_path):
    path = write_config(tmp_path, BASE, """
        analytics:
          footfall: {enabled: true, lines: [entrance]}
          gender: {enabled: true, backend: deepface}
    """)
    with pytest.raises(ConfigError, match="unknown backend"):
        load_config(path, environ={})


def test_ccms_enabled_without_credentials_is_rejected(tmp_path):
    path = write_config(tmp_path, BASE, FOOTFALL, """
        sinks:
          ccms: {enabled: true}
    """)
    with pytest.raises(ConfigError, match="api_key"):
        load_config(path, environ={})


def test_all_cross_reference_errors_reported_together(tmp_path):
    """One run should list every problem, not stop at the first."""
    path = write_config(tmp_path, BASE, """
        analytics:
          footfall: {enabled: true, lines: [nope]}
          viewing_zone: {enabled: true, zones: [nowhere]}
          parking: {enabled: true}
    """)
    with pytest.raises(ConfigError) as excinfo:
        load_config(path, environ={})

    message = str(excinfo.value)
    assert "unknown line 'nope'" in message
    assert "unknown zone 'nowhere'" in message
    assert "parking is enabled but lists no bays" in message


# ==================================================================
# LAYERING
# ==================================================================


def test_env_overrides_beat_file(tmp_path):
    path = write_config(tmp_path, BASE, FOOTFALL)
    config, report = load_config(
        path,
        environ={
            "PIXELSPOT__ANALYTICS__FOOTFALL__ENABLED": "false",
            "PIXELSPOT__RUNTIME__DEVICE": "cuda:0",
        },
    )

    assert config.analytics.footfall.enabled is False
    assert config.runtime.device == "cuda:0"
    assert len(report.env_keys) == 2


def test_cli_overrides_beat_env(tmp_path):
    path = write_config(tmp_path, BASE, FOOTFALL)
    config, _ = load_config(
        path,
        cli_overrides={"runtime": {"device": "cpu"}},
        environ={"PIXELSPOT__RUNTIME__DEVICE": "cuda:0"},
    )
    assert config.runtime.device == "cpu"


def test_env_values_are_typed_not_strings(tmp_path):
    path = write_config(tmp_path, BASE, FOOTFALL)
    config, _ = load_config(
        path,
        environ={
            "PIXELSPOT__RUNTIME__TARGET_FPS": "7.5",
            "PIXELSPOT__PERCEPTION__DETECTOR__CLASSES": '["person","bus"]',
        },
    )
    assert config.runtime.target_fps == 7.5
    assert config.perception.detector.classes == ["person", "bus"]


def test_malformed_env_override_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="malformed environment override"):
        load_config(write_config(tmp_path, BASE, FOOTFALL), environ={"PIXELSPOT__": "x"})


def test_env_typo_still_rejected_by_schema(tmp_path):
    """An override path that does not exist must not be silently ignored."""
    with pytest.raises(ConfigError, match="footfal"):
        load_config(
            write_config(tmp_path, BASE, FOOTFALL),
            environ={"PIXELSPOT__ANALYTICS__FOOTFAL__ENABLED": "true"},
        )


def test_deep_merge_replaces_lists_and_merges_dicts():
    base = {"a": {"x": 1, "y": 2}, "list": [1, 2, 3]}
    override = {"a": {"y": 99}, "list": [7]}
    assert deep_merge(base, override) == {"a": {"x": 1, "y": 99}, "list": [7]}


def test_env_overrides_builds_nested_dict():
    overrides, keys = env_overrides({"PIXELSPOT__SINKS__CCMS__BATCH_SIZE": "10"})
    assert overrides == {"sinks": {"ccms": {"batch_size": 10}}}
    assert keys == ["PIXELSPOT__SINKS__CCMS__BATCH_SIZE"]


# ==================================================================
# VARIABLE EXPANSION
# ==================================================================


def test_variable_expansion_uses_environment():
    result = expand_variables({"url": "${CCMS_URL}"}, {"CCMS_URL": "https://x.test"})
    assert result == {"url": "https://x.test"}


def test_variable_expansion_falls_back_to_default():
    assert expand_variables("${MISSING:-fallback}", {}) == "fallback"


def test_unset_variable_without_default_becomes_none():
    """An absent optional setting must stay absent, not become an empty string."""
    assert expand_variables("${MISSING}", {}) is None


def test_secrets_are_not_read_from_the_config_file(tmp_path):
    path = write_config(tmp_path, BASE, FOOTFALL, """
        sinks:
          ccms:
            enabled: true
            base_url: ${CCMS_URL}
            api_key: ${CCMS_API_KEY}
            device_id: ${DEVICE_ID}
    """)
    config, _ = load_config(
        path,
        environ={
            "CCMS_URL": "https://ccms.test",
            "CCMS_API_KEY": "secret-value",
            "DEVICE_ID": "device-42",
        },
    )
    assert config.sinks.ccms.base_url == "https://ccms.test"
    assert config.sinks.ccms.device_id == "device-42"


# ==================================================================
# FILE HANDLING
# ==================================================================


def test_missing_config_file_reports_the_path(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml", environ={})


def test_invalid_yaml_is_reported_as_config_error(tmp_path):
    path = write_config(tmp_path, "source: {type: file, uri: a.mp4\n  bad indent\n")
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_config(path, environ={})


def test_non_mapping_config_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="mapping"):
        load_config(write_config(tmp_path, "- just\n- a\n- list\n"), environ={})


# ==================================================================
# HELPERS
# ==================================================================


@pytest.mark.parametrize(
    "text, seconds",
    [("30s", 30), ("15m", 900), ("1h", 3600), ("2d", 172800)],
)
def test_parse_duration(text, seconds):
    assert parse_duration(text) == seconds


def test_parse_duration_rejects_garbage():
    with pytest.raises(ValueError, match="invalid duration"):
        parse_duration("15x")


def test_schema_validates_from_a_plain_dict():
    """The schema must be usable without touching the filesystem."""
    config = PixelSpotConfig.model_validate({"source": {"type": "file", "uri": "a.mp4"}})
    assert config.enabled_features() == []


def test_schema_rejects_missing_required_source():
    with pytest.raises(ValidationError, match="source"):
        PixelSpotConfig.model_validate({})
