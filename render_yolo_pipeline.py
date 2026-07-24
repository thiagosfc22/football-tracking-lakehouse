"""Pass 2 — draw tracked YOLO boxes and the ingestion pipeline overlay.

Landscape (1920x1080) treatment:
  - one colour per track id, corner-accented bounding boxes with label chips
  - a record particle per frame, falling from a detection onto the conveyor
    line and running right into the mark at the end, which pulses
  - live counter of ingested rows (one row per detection per frame)

    python render_yolo_pipeline.py clip_60s.mp4 detections.npz OUT.mp4 \
        --logo mark.png --source-label "YOUTUBE sO1J71HS3pA"
"""

import argparse
import os
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------- palette ---
RED = (33, 54, 255)        # BGR — #FF3621 Databricks Lakehouse red
INK = (20, 14, 11)         # BGR — #0B0E14
PAPER = (227, 230, 232)    # BGR — #E8E6E3

# Track colours — vivid, mutually distinct, and none of them Lakehouse red:
# red stays reserved for the pipeline accent so the brand reads instantly.
TRACK_COLORS = [
    (255, 229, 0),     # cyan
    (10, 214, 255),    # yellow
    (255, 132, 10),    # blue
    (88, 209, 48),     # green
    (168, 45, 255),    # magenta
    (10, 159, 255),    # orange
    (242, 90, 191),    # purple
    (224, 200, 64),    # teal
    (130, 100, 255),   # pink
    (94, 92, 230),     # indigo-ish
    (207, 212, 102),   # mint
    (60, 220, 180),    # lime
]
BALL_COLOR = (255, 255, 255)


def _font(env_var, candidates, label):
    override = os.environ.get(env_var)
    if override:
        return override
    for path in candidates:
        if Path(path).exists():
            return path
    raise SystemExit(
        f"no {label} font found. Set {env_var} to a .ttf, or install one of:\n  "
        + "\n  ".join(candidates))


FONT_DISPLAY = _font("POSE_FONT_DISPLAY", [
    "/System/Library/Fonts/Supplemental/DIN Condensed Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf",
], "condensed display")

FONT_MONO = _font("POSE_FONT_MONO", [
    "/System/Library/Fonts/SFNSMono.ttf",
    "/System/Library/Fonts/Menlo.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
], "monospace")

FONT_WORDMARK = _font("POSE_FONT_WORDMARK", [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
], "wordmark sans")

# --------------------------------------------------------------- geometry ---
LINE_Y = 1002              # the conveyor
LINE_X0 = 80
ICON_W = 118
ICON_CX = 1500             # particles are absorbed at the icon centre
WORDMARK_GAP = 26


class Particle:
    """A detection row travelling from its bounding box to the lakehouse."""

    __slots__ = ("x0", "y0", "xr", "age", "life_a", "life_b", "trail", "done")

    def __init__(self, x0, y0, xr, life_a, life_b):
        self.x0, self.y0, self.xr = x0, y0, xr
        self.age = 0.0
        self.life_a, self.life_b = life_a, life_b
        self.trail = []
        self.done = False

    def step(self, dt, x_dest):
        self.age += dt
        if self.age < self.life_a:
            t = self.age / self.life_a
            u = 1 - t
            px = u * u * self.x0 + 2 * u * t * self.x0 + t * t * self.xr
            py = u * u * self.y0 + 2 * u * t * LINE_Y + t * t * LINE_Y
        else:
            t = (self.age - self.life_a) / self.life_b
            if t >= 1.0:
                self.done = True
                return None
            px = self.xr + (x_dest - self.xr) * t
            py = LINE_Y
        pos = (px, py)
        self.trail.append(pos)
        if len(self.trail) > 4:
            self.trail.pop(0)
        return pos


def load_logo(path, width):
    art = Image.open(path).convert("RGBA")
    h = round(art.height * width / art.width)
    art = art.resize((width, h), Image.LANCZOS)
    rgba = np.array(art)
    bgr = rgba[:, :, [2, 1, 0]].astype(np.float32)
    alpha = (rgba[:, :, 3:4].astype(np.float32)) / 255.0
    return bgr, alpha


def alpha_paste(canvas, bgr, alpha, x, y):
    h, w = alpha.shape[:2]
    H, W = canvas.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x0 >= x1 or y0 >= y1:
        return
    sub_a = alpha[y0 - y:y1 - y, x0 - x:x1 - x]
    sub_c = bgr[y0 - y:y1 - y, x0 - x:x1 - x]
    region = canvas[y0:y1, x0:x1].astype(np.float32)
    canvas[y0:y1, x0:x1] = (region * (1 - sub_a) + sub_c * sub_a).astype(np.uint8)


def base_gradient(h, w):
    """Alpha ramp that sinks the footage under the bottom HUD band."""
    ramp = np.zeros((h, w, 1), np.float32)
    fade_top, fade_end, peak = 856, 960, 0.93
    ys = np.arange(h, dtype=np.float32)
    t = np.clip((ys - fade_top) / (fade_end - fade_top), 0, 1)
    ramp[:, :, 0] = (t * t * (3 - 2 * t) * peak)[:, None]  # smoothstep
    return ramp


def build_hud_text(w, h, source_label, dims_label, wordmark_text):
    """Static type for the HUD, rendered once with PIL and reused per frame."""
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    display = ImageFont.truetype(FONT_DISPLAY, 42)
    mono = ImageFont.truetype(FONT_MONO, 20)
    small = ImageFont.truetype(FONT_DISPLAY, 27)
    wordmark = ImageFont.truetype(FONT_WORDMARK, 56)

    def tracked(draw, xy, text, font, fill, extra=3.0):
        x, y = xy
        for ch in text:
            draw.text((x, y), ch, font=font, fill=fill)
            x += draw.textlength(ch, font=font) + extra

    tracked(d, (LINE_X0, 878), "OBJECT DETECTION", display, (232, 230, 227, 255))
    tracked(d, (LINE_X0 + 340, 878), "· YOLOV8X + BYTETRACK", display,
            (168, 179, 194, 255))
    d.text((LINE_X0 + 2, 938), f"SOURCE  {source_label}   {dims_label}",
           font=mono, fill=(140, 152, 168, 255))
    tracked(d, (LINE_X0 + 2, 1026), "INGESTED", small, (140, 152, 168, 255), 2.0)
    tracked(d, (LINE_X0 + 322, 1026), "ROWS", small, (100, 110, 124, 255), 2.0)

    # The lockup: wordmark next to the icon (icon pasted per frame, over the
    # pulse layer).
    d.text((ICON_CX + ICON_W // 2 + WORDMARK_GAP, LINE_Y - 36), wordmark_text,
           font=wordmark, fill=(255, 255, 255, 255))
    return np.array(img)


def track_color(tid):
    if tid < 0:
        return TRACK_COLORS[0]
    return TRACK_COLORS[tid % len(TRACK_COLORS)]


def draw_box(canvas, x1, y1, x2, y2, color, label):
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    # Corner accents so the boxes read as instrumentation, not clip-art.
    c = max(8, min(16, (x2 - x1) // 4))
    for cx, cy, dx, dy in ((x1, y1, 1, 1), (x2, y1, -1, 1),
                           (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(canvas, (cx, cy), (cx + dx * c, cy), color, 4, cv2.LINE_AA)
        cv2.line(canvas, (cx, cy), (cx, cy + dy * c), color, 4, cv2.LINE_AA)
    # Label chip above the box, clamped inside the frame.
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
    ly = max(th + 8, y1 - 6)
    cv2.rectangle(canvas, (x1, ly - th - 8), (x1 + tw + 10, ly + 2), color, -1)
    cv2.putText(canvas, label, (x1 + 5, ly - 4), cv2.FONT_HERSHEY_SIMPLEX,
                0.48, (10, 10, 10), 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("det")
    ap.add_argument("dst")
    ap.add_argument("--logo", required=True,
                    help="PNG with alpha for the mark at the end of the "
                         "pipeline (see README)")
    ap.add_argument("--source-label", default="CLIP")
    ap.add_argument("--wordmark", default="lakehouse",
                    help="text drawn next to the mark")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    det = np.load(args.det)["det"]  # frame, id, cls, conf, x1, y1, x2, y2
    by_frame = {}
    for row in det:
        by_frame.setdefault(int(row[0]), []).append(row)

    cap = cv2.VideoCapture(args.src)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dt = 1.0 / fps

    logo_bgr, logo_a = load_logo(args.logo, ICON_W)
    lh, lw = logo_a.shape[:2]
    logo_x = ICON_CX - lw // 2
    logo_y = LINE_Y - lh // 2
    line_x1 = logo_x - 30

    ramp = base_gradient(h, w)
    ink_plate = np.full((h, w, 3), INK, np.float32)
    hud_rgba = build_hud_text(w, h, args.source_label,
                              f"{w}x{h}  {round(fps)} FPS", args.wordmark)
    hud_bgr = hud_rgba[:, :, [2, 1, 0]].astype(np.float32)
    hud_a = hud_rgba[:, :, 3:4].astype(np.float32) / 255.0

    mono_num = ImageFont.truetype(FONT_MONO, 34)

    rng = random.Random(args.seed)
    particles = []
    pulse = 0.0
    records = 0

    writer = cv2.VideoWriter(args.dst, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (w, h))
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rows = by_frame.get(idx, [])
        idx += 1

        # --- boxes, one colour per track ------------------------------------
        for row in rows:
            _, tid, cls, conf, x1, y1, x2, y2 = row
            tid, cls = int(tid), int(cls)
            if cls == 32:
                draw_box(frame, x1, y1, x2, y2, BALL_COLOR, "BALL")
            else:
                draw_box(frame, x1, y1, x2, y2, track_color(tid),
                         f"ID {tid}  {conf:.2f}")

        # --- emit one record from a detection --------------------------------
        records += len(rows)
        if rows:
            row = rows[rng.randrange(len(rows))]
            bx = float(row[4] + row[6]) / 2
            by = float(row[7])          # bottom edge of the box
            if by < LINE_Y - 60:
                particles.append(Particle(
                    bx, by, bx + rng.uniform(20, 90),
                    rng.uniform(0.85, 1.15), rng.uniform(0.55, 0.78)))

        # --- advance particles ------------------------------------------------
        part_layer = np.zeros_like(frame)
        alive = []
        for p in particles:
            pos = p.step(dt, float(ICON_CX))
            if p.done:
                pulse = 1.0
                continue
            alive.append(p)
            for k in range(1, len(p.trail)):
                a0 = (k / len(p.trail)) ** 2 * 0.55
                x1p, y1p = p.trail[k - 1]
                x2p, y2p = p.trail[k]
                col = tuple(int(c * a0) for c in RED)
                cv2.line(part_layer, (int(x1p), int(y1p)), (int(x2p), int(y2p)),
                         col, 1, cv2.LINE_AA)
            cv2.circle(part_layer, (int(pos[0]), int(pos[1])), 4, RED, -1,
                       cv2.LINE_AA)
            cv2.circle(part_layer, (int(pos[0]), int(pos[1])), 1,
                       (200, 220, 255), -1, cv2.LINE_AA)
        particles = alive

        # --- HUD band ---------------------------------------------------------
        out = (frame.astype(np.float32) * (1 - ramp) + ink_plate * ramp).astype(np.uint8)

        cv2.line(out, (LINE_X0, LINE_Y), (line_x1, LINE_Y), (78, 66, 58), 2,
                 cv2.LINE_AA)
        p_halo = cv2.GaussianBlur(part_layer, (17, 17), 0)
        out = cv2.addWeighted(out, 1.0, p_halo, 0.8, 0)
        out = cv2.addWeighted(out, 1.0, part_layer, 0.95, 0)

        # The icon glows continuously — the brand has to carry the frame — and
        # flares a little brighter each time a record lands.
        glow = 0.35 + 0.65 * pulse
        halo_r = np.zeros_like(out)
        cv2.circle(halo_r, (ICON_CX, LINE_Y), 52,
                   tuple(int(c * glow) for c in RED), -1, cv2.LINE_AA)
        halo_r = cv2.GaussianBlur(halo_r, (61, 61), 0)
        out = cv2.addWeighted(out, 1.0, halo_r, 0.55, 0)
        pulse *= 0.84

        out = (out.astype(np.float32) * (1 - hud_a) + hud_bgr * hud_a).astype(np.uint8)
        alpha_paste(out, logo_bgr, logo_a, logo_x, logo_y)

        plate = Image.new("RGBA", (300, 54), (0, 0, 0, 0))
        ImageDraw.Draw(plate).text(
            (0, 0), f"{records:,}".replace(",", "."), font=mono_num,
            fill=(232, 230, 227, 255))
        pa = np.array(plate)
        alpha_paste(out, pa[:, :, [2, 1, 0]].astype(np.float32),
                    pa[:, :, 3:4].astype(np.float32) / 255.0,
                    LINE_X0 + 148, 1020)

        writer.write(out)
        if idx % 100 == 0 or idx == total:
            print(f"  {idx}/{total}", flush=True)

    cap.release()
    writer.release()
    print(f"wrote {args.dst}  ({records} rows ingested)")


if __name__ == "__main__":
    main()
