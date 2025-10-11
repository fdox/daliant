#!/usr/bin/env python3
import argparse, os, math
import cv2, numpy as np
from PIL import Image
import torch
from torchvision.ops import nms
from ultralytics import YOLO

def tile_coords(W, H, tile, overlap):
    step = int(tile * (1 - overlap))
    if step <= 0: step = tile
    nx = math.ceil(max(W - tile, 0) / step) + 1
    ny = math.ceil(max(H - tile, 0) / step) + 1
    for iy in range(ny):
        for ix in range(nx):
            x0 = ix * step
            y0 = iy * step
            x1 = min(x0 + tile, W)
            y1 = min(y0 + tile, H)
            yield (x0, y0, x1, y1)

def draw_dets(img_bgr, dets, names, thickness=2):
    for (x1, y1, x2, y2, conf, cls) in dets:
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0,255,0), thickness)
        label = f"{names[int(cls)]} {conf:.2f}"
        cv2.putText(img_bgr, label, (x1, max(10,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

def to_yolo_xywhn(dets, W, H):
    # convert xyxy (pixels) to YOLO txt lines: class cx cy w h (normalized)
    lines = []
    for (x1, y1, x2, y2, conf, cls) in dets:
        cx = ((x1 + x2) / 2.0) / W
        cy = ((y1 + y2) / 2.0) / H
        w  = (x2 - x1) / W
        h  = (y2 - y1) / H
        lines.append(f"{int(cls)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {conf:.4f}")
    return "\n".join(lines)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="runs/fullpage_pred")
    ap.add_argument("--tile", type=int, default=1536)
    ap.add_argument("--overlap", type=float, default=0.20)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--device", default="mps")  # "cpu" or "mps"
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Load model
    model = YOLO(args.weights)
    names = model.names

    # Read full image
    full = Image.open(args.image).convert("RGB")
    W, H = full.size
    full_bgr = cv2.cvtColor(np.array(full), cv2.COLOR_RGB2BGR)

    all_boxes = []
    all_scores = []
    all_classes = []

    # Run tiled inference
    for (x0, y0, x1, y1) in tile_coords(W, H, args.tile, args.overlap):
        tile = full.crop((x0, y0, x1, y1))
        res = model.predict(
            source=np.array(tile),
            device=args.device,
            imgsz=args.tile,
            conf=args.conf,
            iou=args.iou,
            verbose=False
        )[0]

        if res.boxes is None or len(res.boxes) == 0:
            continue

        b = res.boxes
        xyxy = b.xyxy.cpu().numpy()
        conf = b.conf.cpu().numpy()
        cls  = b.cls.cpu().numpy()

        # offset boxes to full-image coords
        xyxy[:, [0,2]] += x0
        xyxy[:, [1,3]] += y0

        all_boxes.append(xyxy)
        all_scores.append(conf)
        all_classes.append(cls)

    if not all_boxes:
        print("No detections found.")
        return

    boxes = np.concatenate(all_boxes, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    classes = np.concatenate(all_classes, axis=0)

    # Run class-wise NMS
    kept = []
    for c in np.unique(classes):
        mask = (classes == c)
        b = torch.tensor(boxes[mask], dtype=torch.float32)
        s = torch.tensor(scores[mask], dtype=torch.float32)
        if b.numel() == 0:
            continue
        keep_idx = nms(b, s, args.iou).numpy()
        for i in keep_idx:
            kept.append((*b[i].numpy().tolist(), float(s[i]), float(c)))

    # Draw and save
    draw_dets(full_bgr, kept, names)
    base = os.path.splitext(os.path.basename(args.image))[0]
    out_img = os.path.join(args.out, f"{base}_tiled_pred.jpg")
    out_txt = os.path.join(args.out, f"{base}_tiled_pred.txt")
    cv2.imwrite(out_img, full_bgr)
    with open(out_txt, "w") as f:
        f.write(to_yolo_xywhn(kept, W, H))

    print(f"Saved: {out_img}")
    print(f"Saved: {out_txt}")
    print(f"Detections kept: {len(kept)}")

if __name__ == "__main__":
    main()
