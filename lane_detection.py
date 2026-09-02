"""
Classical OpenCV lane detection: Canny edges -> ROI mask -> probabilistic
Hough transform -> average/extrapolate into a single left and right lane line.
"""
import numpy as np
import cv2


def _region_of_interest(edges):
    h, w = edges.shape[:2]
    mask = np.zeros_like(edges)
    polygon = np.array([[
        (int(w * 0.03), h),
        (int(w * 0.40), int(h * 0.55)),
        (int(w * 0.60), int(h * 0.55)),
        (int(w * 0.97), h),
    ]], dtype=np.int32)
    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(edges, mask)


def _average_slope_intercept(lines, w, h):
    left, right = [], []
    if lines is None:
        return None, None
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1:
            continue
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        if abs(slope) < 0.35:  # drop near-horizontal noise
            continue
        if slope < 0:
            left.append((slope, intercept))
        else:
            right.append((slope, intercept))

    def make_line(fit):
        if not fit:
            return None
        slope, intercept = np.mean(fit, axis=0)
        y1 = h
        y2 = int(h * 0.58)
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        return (x1, y1, x2, y2)

    return make_line(left), make_line(right)


def detect_lanes(frame):
    """Returns (annotated_frame, offset) where offset is the normalized
    horizontal distance of the lane center from the frame center
    (-1 = lane fully left, +1 = lane fully right, None = lanes not found)."""
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    roi = _region_of_interest(edges)

    lines = cv2.HoughLinesP(roi, 2, np.pi / 180, threshold=40,
                             minLineLength=30, maxLineGap=120)
    left, right = _average_slope_intercept(lines, w, h)

    overlay = np.zeros_like(frame)
    for line in (left, right):
        if line is not None:
            x1, y1, x2, y2 = line
            cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 0), 8)

    offset = None
    if left is not None and right is not None:
        lane_center = (left[0] + right[0]) / 2  # bottom-of-frame x for each line
        frame_center = w / 2
        offset = (lane_center - frame_center) / (w / 2)
        mid = np.array([[int(lane_center), h - 5], [int(frame_center), h - 5]])
        cv2.line(overlay, tuple(mid[0]), tuple(mid[1]), (0, 0, 255), 4)

    annotated = cv2.addWeighted(frame, 1.0, overlay, 0.8, 0)
    return annotated, offset
