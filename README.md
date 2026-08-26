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

When demoing from a video file on a CPU-only machine, set
`source.realtime: true` so the recording plays at its own speed — frames
inference cannot keep up with are skipped, exactly as a live camera would
drop them. Leave it `false` (the default) when you want every frame
analysed, e.g. for offline measurement.

## Using an Android phone as the camera

PixelSpot reads network streams (`source.type: http` or `rtsp`), so any app
that turns the phone into an IP camera works. The simplest setup uses the
free **IP Webcam** app:

1. Install *IP Webcam* (by Pavel Khlebovich) from the Play Store.
2. Connect the phone and the computer to the **same Wi-Fi network**.
3. Open the app, scroll to the bottom and tap **Start server**. The screen
   shows an address such as `http://192.168.1.42:8080`.
4. Confirm it works by opening that address in the computer's browser; the
   raw video stream lives at the `/video` path.
5. Point PixelSpot at the stream:

   ```bash
   pixelspot run --source http://192.168.1.42:8080/video
   ```

   or make it permanent in `config/config.yaml`:

   ```yaml
   source:
     type: http
     uri: http://192.168.1.42:8080/video
   ```

Apps that serve RTSP instead (e.g. *RTSP Camera Server*) work the same way
with `type: rtsp` and an `rtsp://...` URI. A third option is **DroidCam**,
whose Windows client installs a virtual webcam driver — the phone then shows
up as an ordinary camera index and runs as `pixelspot run --source 1`
(try 0, 1, 2... to find the right index).

Tips for a stable phone feed:

- **Prefer USB tethering over Wi-Fi** when you can (enable it in the phone's
  hotspot settings): lower latency and no dropouts when the network is busy.
- In IP Webcam's settings, lower the video resolution to 1280x720 — plenty
  for detection, and it halves the network and inference load.
- Lock the phone in **landscape** orientation and disable auto-rotate so the
  frame does not flip mid-run.
- Keep the screen awake (IP Webcam does this while serving) and plug the
  phone in; streaming drains the battery quickly.
- Network hiccups are handled automatically: `source.reconnect` retries with
  backoff, and the frame buffer's `drop_oldest` policy keeps the analytics
  looking at the *current* view rather than drifting behind real time.

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
