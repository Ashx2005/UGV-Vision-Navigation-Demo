"""
UGV Vision Navigation Demo -- end-to-end pipeline.

video in -> lane detection -> obstacle detection -> simple navigation
decision overlay -> annotated video out.

Run against the bundled synthetic footage:
    python main.py

Run against real dashcam footage:
    python main.py --source path/to/real_dashcam.mp4
"""
import argparse
import os
import time

import cv2

from lane_detection import detect_lanes
from obstacle_detection import ColorBlobDetector, MobileNetSSDDetector, draw_detections


def build_detectors(args):
    proto = os.path.join(args.model_dir, "MobileNetSSD_deploy.prototxt")
    model = os.path.join(args.model_dir, "MobileNetSSD_deploy.caffemodel")

    ssd = None
    if args.detector in ("ssd", "both"):
        if os.path.exists(proto) and os.path.exists(model):
            ssd = MobileNetSSDDetector(proto, model)
        else:
            print(f"[warn] SSD model files not found in {args.model_dir}/, "
                  f"skipping pretrained detector")

    color = ColorBlobDetector() if args.detector in ("color", "both") else None
    return ssd, color


def decide_action(offset, detections, w, h):
    """Very simple rule-based decision: obstacle-in-path overrides steering."""
    danger = False
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        cx = (x1 + x2) / 2
        close = y2 > h * 0.75
        centered = w * 0.25 < cx < w * 0.75
        if close and centered:
            danger = True

    lines = []
    if danger:
        lines.append("OBSTACLE AHEAD -- STOP")
    elif detections:
        lines.append(f"{len(detections)} obstacle(s) detected -- caution")

    if offset is None:
        lines.append("LANE NOT DETECTED")
    elif not danger:
        if offset > 0.08:
            lines.append("STEER RIGHT")
        elif offset < -0.08:
            lines.append("STEER LEFT")
        else:
            lines.append("LANE OK -- STRAIGHT")
    return lines


def main():
    parser = argparse.ArgumentParser(description="UGV vision navigation demo")
    parser.add_argument("--source", default="sample_data/road_test.mp4")
    parser.add_argument("--output", default="output/annotated.mp4")
    parser.add_argument("--detector", choices=["ssd", "color", "both"], default="ssd",
                         help="ssd = pretrained MobileNet-SSD only (default; use this for "
                              "real footage), color = classical fallback only (use this for "
                              "the synthetic demo clip, where the SSD model detects nothing "
                              "-- see README), both = run and overlay both")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--display", action="store_true", help="show a live preview window")
    parser.add_argument("--max-frames", type=int, default=0, help="0 = process whole video")
    args = parser.parse_args()

    ssd, color = build_detectors(args)

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source: {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    frame_idx = 0
    total_ssd_dets = 0
    total_color_dets = 0
    t_start = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if args.max_frames and frame_idx > args.max_frames:
            break

        annotated, offset = detect_lanes(frame)

        detections = []
        if ssd is not None:
            d = ssd.detect(frame)
            total_ssd_dets += len(d)
            detections.extend(d)
        if color is not None:
            d = color.detect(frame)
            total_color_dets += len(d)
            detections.extend(d)
        draw_detections(annotated, detections)

        lines = decide_action(offset, detections, w, h)
        for i, text in enumerate(lines):
            text_color = (0, 0, 255) if "STOP" in text else (255, 255, 0)
            y = h - 12 - 22 * (len(lines) - 1 - i)
            cv2.putText(annotated, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, text_color, 2, cv2.LINE_AA)

        writer.write(annotated)
        if args.display:
            cv2.imshow("UGV Vision Navigation Demo", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    elapsed = time.time() - t_start
    cap.release()
    writer.release()
    if args.display:
        cv2.destroyAllWindows()

    print(f"Processed {frame_idx} frames in {elapsed:.1f}s "
          f"({frame_idx / max(elapsed, 1e-6):.1f} fps)")
    print(f"Pretrained-SSD detections total: {total_ssd_dets}, "
          f"color-blob detections total: {total_color_dets}")
    print(f"Annotated video written to: {args.output}")
    os.startfile(os.path.abspath(args.output))


if __name__ == "__main__":
    main()
