# PixelSpot AI Vision

Edge computer-vision audience analytics. PixelSpot watches a camera feed
(file, webcam, RTSP, or HTTP), detects and tracks people and vehicles with
YOLO + ByteTrack, and turns the tracks into privacy-conscious metrics —
footfall counts, zone occupancy, vehicle counts, and more — which it emits to
the console, a JSONL file, or a CCMS backend.

## How it works

```
source ──> detector (YOLO) ──> tracker (ByteTrack) ──> analytics processors ──> aggregator ──> sinks
                                                          footfall, zones,        1m/15m/1h      console
                                                          vehicles, ...           windows        jsonl, ccms
```

Everything is driven by one YAML config. Each analytics capability is a
pluggable processor: `analytics.<name>.enabled` in the config decides what
runs, and `src/pixelspot/analytics/registry.py` maps names to
implementations. Adding a capability is one class plus one registry entry.

## Quickstart

```bash
# 1. Set up the environment
python -m venv .venv
.venv/Scripts/activate          # Windows; use .venv/bin/activate elsewhere
pip install -r requirements.txt
pip install -e .

# 2. Create your local config (git-ignored; the example lists every setting)
cp config/config.example.yaml config/config.yaml

# 3. Check the config without opening the camera
pixelspot validate-config

# 4. Run
pixelspot run                       # source from config.yaml
pixelspot run --source 0            # webcam
pixelspot run --headless            # no preview window (e.g. on a device)
```

Model weights are not committed. Place them in `models/` (the detector
default is `models/yolo11n.pt`; Ultralytics downloads it on first use if
missing).

Settings layer, lowest priority first:
`built-in defaults < config.yaml < config/sites/<name>.yaml < environment < CLI`.
Environment overrides use `__` between levels, e.g.
`PIXELSPOT__RUNTIME__DEVICE=cuda:0`.

## Capabilities

All 16 capabilities are implemented:

- **Counting & presence** — footfall (line crossing), viewing_zone occupancy, vehicle counting/classification
- **Derived from tracks + zones + time** — dwell time, crowd_density (people/m²), traffic_direction, queue detection, audience_flow (zone-to-zone), heatmap, parking bays, anomaly rules
- **Head pose** (`perception.enrichment.head_pose`) — attention (sustained gaze at a screen), screen_visibility
- **Face attributes** (`perception.enrichment.face` + `backend: opencv_dnn`) — gender, age buckets, mood; classified via OpenCV DNN models (YuNet, GoogleNet age/gender, FER+) with per-person vote-over-time smoothing

## Privacy

Face-derived attributes (age/gender/mood) are biometric processing under
GDPR and the DPDP Act. Defaults are restrictive on purpose: no crops stored,
faces blurred in output, aggregate-only metrics. See the `privacy` section
of `config/config.example.yaml`.

## Project layout

```
config/
  config.example.yaml   tracked reference config (every setting + default)
  config.yaml           your local config (git-ignored)
src/pixelspot/
  cli.py, app.py        entry point and pipeline wiring
  settings/             schema, layered loader, startup banner
  source.py             capture with reconnect + frame buffering
  detection/, tracking/ YOLO detector, ByteTrack tracker
  geometry.py           normalized zones / lines / screens
  analytics/            processors + registry (one class per capability)
  aggregation/          windowed metrics (1m / 15m / 1h)
  sinks/                console, JSONL, CCMS (with offline spooling)
  render.py             preview overlay
tests/                  pytest suite
```

## Tests

```bash
python -m pytest tests -q
```
