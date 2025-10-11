#!/usr/bin/env python3
import argparse, os
from PIL import Image
from math import ceil

def tile_image(path, out_dir, tile, overlap):
    im = Image.open(path).convert("RGB")
    W, H = im.size
    step = int(tile * (1 - overlap))
    if step <= 0:
        step = tile
    nx = ceil(max(W - tile, 0) / step) + 1
    ny = ceil(max(H - tile, 0) / step) + 1
    base = os.path.splitext(os.path.basename(path))[0]
    for iy in range(ny):
        for ix in range(nx):
            x0 = ix * step
            y0 = iy * step
            x1 = min(x0 + tile, W)
            y1 = min(y0 + tile, H)
            crop = im.crop((x0, y0, x1, y1))
            tag = f"{base}_x{ix:02d}_y{iy:02d}.png"
            crop.save(os.path.join(out_dir, tag))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tile", type=int, default=1536)
    ap.add_argument("--overlap", type=float, default=0.15, help="0..0.9 fraction")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for fname in os.listdir(args.in_dir):
        if not fname.lower().endswith((".png",".jpg",".jpeg")):
            continue
        path = os.path.join(args.in_dir, fname)
        tile_image(path, args.out_dir, args.tile, args.overlap)
        print("Tiled", fname)

if __name__ == "__main__":
    main()
