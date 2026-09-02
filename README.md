# UGV Vision Navigation Demo

3-hour MVP: proves the smallest end-to-end flow for a UGV perception stack --
video in, lane + obstacle detection, a simple navigation decision, video out.
No physical vehicle required.

## Pipeline

```
video frame
   |-- lane_detection.py     Canny edges -> ROI mask -> Hough transform ->
   |                         averaged left/right lane lines -> lane-center offset
   |-- obstacle_detection.py Pretrained MobileNet-SSD (OpenCV DNN) for real
   |                         footage, + a classical color/contour fallback
   |                         (see "Why two detectors" below)
   v
decide_action() in main.py  -> "STEER LEFT/RIGHT", "LANE OK", or
                                "OBSTACLE AHEAD -- STOP" overlay
   v
output/annotated.mp4
```

## Run it

Default detector is the pretrained SSD model, so point it at real footage:

```
python main.py --source path\to\real_dashcam.mp4
```

Validated against `sample_data/real_dashcam.webm` (public-domain highway
dashcam clip, see below) -- 437 real `car` detections across 902 frames, zero
false positives.

Against the bundled synthetic clip, use the color fallback instead (the SSD
model detects nothing on procedurally-drawn scenes -- see "Why two obstacle
detectors"):

```
python generate_test_video.py           # only needed once, makes sample_data/road_test.mp4
python main.py --detector color         # writes output/annotated.mp4
```

Other flags: `--detector {ssd,color,both}` (default `ssd`), `--display` (live
preview window), `--max-frames N` (quick smoke test).

## Why two obstacle detectors

There's no real dashcam footage on this machine, so `generate_test_video.py`
renders a synthetic road scene (sky/grass/road, dashed lane markings, an
approaching car sprite) directly with OpenCV drawing calls. The pretrained
MobileNet-SSD model (`models/`, from the standard 21-class Caffe MobileNet-SSD)
is wired in exactly as it would be for real footage, but deep detectors
trained on real photos generally don't fire on procedurally-drawn shapes --
there's no real-world texture/gradient statistics for them to key off (the
sim-to-real gap). Confirmed empirically: 0 SSD detections on the synthetic
clip.

So a classical HSV color-blob fallback (`ColorBlobDetector`) detects the
synthetic red car by color/contour, which is what actually draws the visible
obstacle boxes when testing against the synthetic clip.

On real footage the SSD model does detect person/car/bus/bicycle/etc, which
is why it's the default -- but the color-blob fallback should NOT be run
alongside it there: real scenes have unrelated red objects (signage,
taillights, distant cars) that it flags as false-positive "obstacles". Use
`--detector ssd` (the default) for real footage, `--detector color` only for
the synthetic clip.

## Files

- `generate_test_video.py` -- synthetic dashcam-style footage generator
- `lane_detection.py` -- classical lane-line detector
- `obstacle_detection.py` -- MobileNetSSDDetector (pretrained) + ColorBlobDetector (fallback)
- `main.py` -- end-to-end pipeline, decision logic, CLI
- `models/` -- pretrained MobileNet-SSD weights (prototxt + caffemodel)
- `sample_data/` -- `road_test.mp4` (synthetic) and `real_dashcam.webm` (real,
  public domain, from [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Multiple_cars_rear-end_collision_on_highway.webm))
- `output/` -- annotated output videos

## Next steps beyond the 3-hour MVP

- Replace the color-blob fallback with a model that actually generalizes to
  synthetic scenes (e.g. fine-tune, or render more photorealistic test data).
- Smooth the lane offset over frames (moving average) instead of per-frame.
- Turn the on/off "STOP" decision into a distance/speed-aware controller.
