import argparse, os, shutil
from pathlib import Path
import cv2, numpy as np
from PIL import Image

def load_yolo_polys(label_path, w, h):
    polys = []
    if not label_path.exists() or label_path.stat().st_size == 0:
        return polys
    for line in label_path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 3: 
            continue
        coords = list(map(float, parts[1:]))
        pts = np.array(coords, dtype=np.float32).reshape(-1,2)
        pts[:,0] *= w; pts[:,1] *= h
        polys.append(pts.astype(np.int32))
    return polys

def write_yolo_polys(label_path, contours, w, h, min_area=12.0, approx_eps=1.5):
    lines = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        approx = cv2.approxPolyDP(cnt, approx_eps, True).reshape(-1,2).astype(np.float32)
        approx[:,0] /= float(w); approx[:,1] /= float(h)
        coords = " ".join([f"{x:.6f} {y:.6f}" for x,y in approx])
        lines.append("0 " + coords)
    label_path.write_text("\n".join(lines))

def symlink_or_copy(src_dir: Path, dst_dir: Path):
    if dst_dir.exists():
        if dst_dir.is_symlink() or dst_dir.is_file():
            dst_dir.unlink()
        else:
            shutil.rmtree(dst_dir)
    dst_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(src_dir.resolve(), dst_dir)
    except OSError:
        # Fallback: copy tree if symlink not allowed
        shutil.copytree(src_dir, dst_dir)

def process_split(src_img_dir: Path, src_lbl_dir: Path, dst_img_dir: Path, dst_lbl_dir: Path, r_px: int, min_area: float, approx_eps: float):
    # images: symlink/copy
    symlink_or_copy(src_img_dir, dst_img_dir)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    img_paths = sorted([p for p in src_img_dir.iterdir() if p.suffix.lower() in (".jpg",".jpeg",".png",".bmp",".tif",".tiff")])
    n_imgs = n_empty = n_written = 0

    for ip in img_paths:
        n_imgs += 1
        with Image.open(ip) as im:
            w,h = im.size
        src_lp = src_lbl_dir / (ip.stem + ".txt")
        dst_lp = dst_lbl_dir / (ip.stem + ".txt")

        polys = load_yolo_polys(src_lp, w, h)
        if not polys:
            dst_lp.write_text("")  # keep true negatives
            n_empty += 1
            continue

        # Rasterize polygons to mask
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, polys, 255)

        # Dilate (thicken) by r_px with elliptical kernel
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r_px+1, 2*r_px+1))
        mask = cv2.dilate(mask, k, iterations=1)

        # Contours back to polygons
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        write_yolo_polys(dst_lp, contours, w, h, min_area=min_area, approx_eps=approx_eps)
        n_written += 1

    print(f"[split @ {src_img_dir.parent.name}] images: {n_imgs} | thickened: {n_written} | empties: {n_empty}")

def main():
    ap = argparse.ArgumentParser(description="Thicken YOLO-seg polygons by rasterize→dilate→vectorize.")
    ap.add_argument("--src_root", default="datasets/wire", help="Root with train/val/images|labels")
    ap.add_argument("--dst_root", default="datasets/wire_thick", help="Output root for thickened dataset")
    ap.add_argument("--radius_px", type=int, default=2, help="Dilation radius in pixels (train resolution)")
    ap.add_argument("--min_area", type=float, default=12.0, help="Contour area threshold to keep")
    ap.add_argument("--approx_eps", type=float, default=1.5, help="Polygon simplification epsilon (pixels)")
    args = ap.parse_args()

    src = Path(args.src_root)
    dst = Path(args.dst_root)
    for split in ["train", "val"]:
        process_split(
            src / split / "images", src / split / "labels",
            dst / split / "images", dst / split / "labels",
            r_px=args.radius_px, min_area=args.min_area, approx_eps=args.approx_eps
        )
    print(f"Thickened dataset written to: {dst}")

if __name__ == "__main__":
    main()
