"""Pass 1 — YOLOv8x + ByteTrack over the clip, cache every detection.

Each detection becomes one row: frame, track_id, cls, conf, x1, y1, x2, y2.
Saved twice: NPZ for the renderer, CSV as the actual artifact a pipeline
would ingest.

    python extract_track.py clip_60s.mp4 detections.npz detections.csv
"""

import argparse
import csv

import numpy as np
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("out_npz")
ap.add_argument("out_csv")
ap.add_argument("--model", default="yolov8x.pt",
                help="ultralytics checkpoint; downloaded on first use")
ap.add_argument("--device", default=None,
                help="mps / cuda / cpu; default lets ultralytics pick")
args = ap.parse_args()

model = YOLO(args.model)
rows = []
stream = model.track(args.src, stream=True, tracker="bytetrack.yaml",
                     classes=[0, 32], conf=0.3, iou=0.5, imgsz=1280,
                     device=args.device, verbose=False)
for i, r in enumerate(stream):
    b = r.boxes
    if b is not None and len(b):
        ids = b.id.cpu().numpy() if b.id is not None else np.full(len(b), -1.0)
        cls = b.cls.cpu().numpy()
        conf = b.conf.cpu().numpy()
        xyxy = b.xyxy.cpu().numpy()
        for k in range(len(b)):
            rows.append((i, int(ids[k]), int(cls[k]), round(float(conf[k]), 4),
                         *[round(float(v), 1) for v in xyxy[k]]))
    if i % 100 == 0:
        print(f"  frame {i}  rows {len(rows)}", flush=True)

arr = np.array(rows, np.float32)
np.savez_compressed(args.out_npz, det=arr)
with open(args.out_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["frame", "track_id", "cls", "conf", "x1", "y1", "x2", "y2"])
    w.writerows(rows)
print(f"saved {arr.shape[0]} rows, {len(set(int(r[1]) for r in rows))} tracks")
