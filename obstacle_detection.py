"""
Obstacle detection.

- MobileNetSSDDetector: real pretrained model (Caffe MobileNet-SSD, 21 COCO/VOC
  classes) run through OpenCV's DNN module. This is what a real UGV would use
  on real dashcam footage.
- ColorBlobDetector: a classical HSV/contour fallback. Deep detectors trained
  on real photos generally do not fire on procedurally-drawn synthetic scenes
  (no learned real-world texture), so this fallback guarantees visible
  obstacle boxes when running against the synthetic demo video. Swap it out
  for the SSD detector's output when pointing the pipeline at real footage.
"""
import numpy as np
import cv2

SSD_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]
RELEVANT_LABELS = {
    "person", "car", "bus", "bicycle", "motorbike", "cat", "dog", "cow",
    "horse", "sheep",
}


class MobileNetSSDDetector:
    def __init__(self, prototxt_path, model_path, conf_threshold=0.4):
        self.net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
        self.conf_threshold = conf_threshold

    def detect(self, frame):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        self.net.setInput(blob)
        detections = self.net.forward()

        results = []
        for i in range(detections.shape[2]):
            conf = float(detections[0, 0, i, 2])
            if conf < self.conf_threshold:
                continue
            idx = int(detections[0, 0, i, 1])
            label = SSD_CLASSES[idx] if idx < len(SSD_CLASSES) else str(idx)
            if label not in RELEVANT_LABELS:
                continue
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            results.append({"box": (x1, y1, x2, y2), "label": label, "conf": conf})
        return results


class ColorBlobDetector:
    """Finds red-ish blobs (matches the synthetic obstacle cars' color)."""

    def __init__(self, min_area=150):
        self.min_area = min_area

    def detect(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0, 100, 60), (10, 255, 255)) | \
            cv2.inRange(hsv, (170, 100, 60), (180, 255, 255))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        results = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            results.append({"box": (x, y, x + w, y + h), "label": "obstacle", "conf": 1.0})
        return results


def draw_detections(frame, detections):
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
        text = f"{det['label']} {det['conf']:.2f}"
        cv2.putText(frame, text, (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1, cv2.LINE_AA)
    return frame
