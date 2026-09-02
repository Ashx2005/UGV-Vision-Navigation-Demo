"""
Generates a synthetic dashcam-style road video for the UGV vision demo.

No real footage is available on this machine, so this renders a simple
perspective road scene (sky, grass, road, dashed lane lines, and an
approaching "car" obstacle) directly with OpenCV drawing primitives.
The pipeline in main.py works the same way on real dashcam footage --
just point --source at a real video file instead.
"""
import math
import cv2
import numpy as np

WIDTH, HEIGHT = 960, 540
FPS = 30
DURATION_SEC = 20
HORIZON_Y = int(HEIGHT * 0.55)
ROAD_HALF_WIDTH_BOTTOM = int(WIDTH * 0.42)
ROAD_HALF_WIDTH_TOP = 18
DRIFT_AMPLITUDE = 70
DRIFT_PERIOD_SEC = 8


def lerp(a, b, t):
    return a + (b - a) * t


def vanishing_point_x(t):
    return WIDTH / 2 + DRIFT_AMPLITUDE * math.sin(2 * math.pi * t / DRIFT_PERIOD_SEC)


def road_edges_at(y, t):
    """Return (left_x, right_x) of the road at a given screen row y."""
    span = HEIGHT - HORIZON_Y
    frac = 0.0 if span <= 0 else (y - HORIZON_Y) / span
    frac = max(0.0, min(1.0, frac))
    vp_x = vanishing_point_x(t)
    half_w = lerp(ROAD_HALF_WIDTH_TOP, ROAD_HALF_WIDTH_BOTTOM, frac)
    return vp_x - half_w, vp_x + half_w


def draw_background(frame, t):
    frame[0:HORIZON_Y, :] = (235, 206, 135)  # sky (BGR)
    frame[HORIZON_Y:HEIGHT, :] = (60, 130, 60)  # grass

    vp_x = vanishing_point_x(t)
    bl_x, br_x = road_edges_at(HEIGHT, t)
    tl_x, tr_x = road_edges_at(HORIZON_Y, t)
    road_poly = np.array([
        [tl_x, HORIZON_Y], [tr_x, HORIZON_Y],
        [br_x, HEIGHT], [bl_x, HEIGHT],
    ], dtype=np.int32)
    cv2.fillPoly(frame, [road_poly], (70, 70, 70))
    return vp_x


def draw_lane_lines(frame, t):
    """Dashed white center-ish lane lines with a scrolling dash pattern."""
    scroll = (t * 220) % 60  # px/sec dash scroll speed, wraps every 60px
    n_steps = 40
    for lane_frac in (0.18, 0.82):  # two lane boundary lines inside road
        for i in range(n_steps):
            y0 = HORIZON_Y + (HEIGHT - HORIZON_Y) * (i / n_steps)
            y1 = HORIZON_Y + (HEIGHT - HORIZON_Y) * ((i + 1) / n_steps)
            dash_pos = (y0 + scroll) % 60
            if dash_pos > 30:
                continue  # gap
            lx0, rx0 = road_edges_at(y0, t)
            lx1, rx1 = road_edges_at(y1, t)
            x0 = lerp(lx0, rx0, lane_frac)
            x1 = lerp(lx1, rx1, lane_frac)
            thickness = max(1, int(lerp(1, 6, (y0 - HORIZON_Y) / (HEIGHT - HORIZON_Y))))
            cv2.line(frame, (int(x0), int(y0)), (int(x1), int(y1)), (255, 255, 255), thickness)


def draw_car(frame, cx, cy, scale, color=(30, 30, 200)):
    """Simple procedural car sprite, sized by `scale` (bigger = closer)."""
    w = max(4, int(46 * scale))
    h = max(3, int(28 * scale))
    x0, y0 = int(cx - w / 2), int(cy - h / 2)
    x1, y1 = int(cx + w / 2), int(cy + h / 2)
    cv2.rectangle(frame, (x0, y0), (x1, y1), color, -1)
    # windshield
    wx0, wy0 = int(cx - w * 0.28), int(y0 - h * 0.35)
    wx1, wy1 = int(cx + w * 0.28), y0
    cv2.rectangle(frame, (wx0, max(0, wy0)), (wx1, y0), (200, 220, 230), -1)
    # headlights
    r = max(1, int(3 * scale))
    cv2.circle(frame, (x0 + r, y1 - r), r, (0, 240, 255), -1)
    cv2.circle(frame, (x1 - r, y1 - r), r, (0, 240, 255), -1)


class ObstacleCar:
    def __init__(self, lane_offset, phase):
        self.lane_offset = lane_offset  # -1 = left lane, 0 = ahead, 1 = right lane
        self.phase = phase  # 0..1 progress from far to near

    def step(self, dt, speed):
        self.phase += dt * speed
        return self.phase < 1.0

    def draw(self, frame, t):
        y = lerp(HORIZON_Y + 5, HEIGHT - 20, self.phase)
        lx, rx = road_edges_at(y, t)
        lane_center = lerp(lx, rx, 0.5)
        lane_width = rx - lx
        cx = lane_center + self.lane_offset * lane_width * 0.28
        scale = lerp(0.15, 2.2, self.phase)
        draw_car(frame, cx, y, scale)


def main():
    out_path = "sample_data/road_test.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, FPS, (WIDTH, HEIGHT))

    n_frames = FPS * DURATION_SEC
    rng = np.random.default_rng(7)
    obstacles = []
    spawn_schedule = sorted(rng.uniform(0, DURATION_SEC - 3, size=4))
    spawn_idx = 0

    for i in range(n_frames):
        t = i / FPS
        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        draw_background(frame, t)
        draw_lane_lines(frame, t)

        if spawn_idx < len(spawn_schedule) and t >= spawn_schedule[spawn_idx]:
            lane_offset = rng.choice([-1, 0, 1])
            obstacles.append(ObstacleCar(lane_offset, phase=0.0))
            spawn_idx += 1

        alive = []
        for obs in obstacles:
            if obs.step(1.0 / FPS, speed=0.22):
                obs.draw(frame, t)
                alive.append(obs)
        obstacles = alive

        cv2.putText(frame, f"SIM DASHCAM  t={t:5.1f}s", (12, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(frame)

    writer.release()
    print(f"Wrote {n_frames} frames ({DURATION_SEC}s @ {FPS}fps) to {out_path}")


if __name__ == "__main__":
    main()
