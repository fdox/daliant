import argparse, sys
from pathlib import Path
import numpy as np, cv2
from PIL import Image, ImageOps

def tile_positions(w, h, tile=1536, overlap=0.30):
    stride = max(1, int(tile * (1.0 - overlap)))
    xs = list(range(0, max(1, w - tile + 1), stride))
    ys = list(range(0, max(1, h - tile + 1), stride))
    if xs[-1] != max(0, w - tile): xs.append(max(0, w - tile))
    if ys[-1] != max(0, h - tile): ys.append(max(0, h - tile))
    return [(x, y) for y in ys for x in xs]

def crop_with_pad(img_bgr, x, y, tile):
    h, w = img_bgr.shape[:2]
    crop = img_bgr[y:min(y+tile, h), x:min(x+tile, w)]
    if crop.shape[0] == tile and crop.shape[1] == tile:
        return crop
    # pad to tile with white background
    canvas = np.full((tile, tile, 3), 255, dtype=np.uint8)
    canvas[:crop.shape[0], :crop.shape[1]] = crop
    return canvas

def main():
    ap = argparse.ArgumentParser(description="Full-PDF wire segmentation: tile → predict → stitch")
    ap.add_argument("--pdf", required=True, help="Path to PDF, e.g., plans_raw/Electrical_Plan_Test3.pdf")
    ap.add_argument("--weights", required=True, help="Path to YOLO-seg weights .pt")
    ap.add_argument("--out", default="runs/fullpdf_wire", help="Output folder")
    ap.add_argument("--tile", type=int, default=1536)
    ap.add_argument("--overlap", type=float, default=0.30)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--device", default="mps", help="mps|cpu|0,...")
    ap.add_argument("--close_px", type=int, default=1, help="post close radius (px) to bridge tile seams; 0=off")
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except Exception as e:
        print("ERROR: ultralytics not available in this env.", file=sys.stderr); raise
    try:
        from pdf2image import convert_from_path
    except Exception:
        print("ERROR: pdf2image not available. Install with `pip install pdf2image pillow` and Poppler.", file=sys.stderr); raise

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = Path(args.pdf)
    pages = convert_from_path(str(pdf_path), dpi=args.dpi)
    model = YOLO(str(args.weights))

    for i, pil_page in enumerate(pages, start=1):
        page_id = f"{pdf_path.stem}_p{i:02d}"
        rgb = np.array(pil_page.convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        H, W = bgr.shape[:2]

        page_mask = np.zeros((H, W), dtype=np.uint8)

        for (x, y) in tile_positions(W, H, tile=args.tile, overlap=args.overlap):
            tile_bgr = crop_with_pad(bgr, x, y, args.tile)
            preds = model.predict(
                source=[tile_bgr], imgsz=args.tile, conf=args.conf, iou=args.iou,
                device=args.device, verbose=False, retina_masks=True, stream=False, save=False
            )
            if not preds: 
                continue
            r = preds[0]
            if getattr(r, "masks", None) is None:
                continue
            confs = r.boxes.conf.cpu().numpy().tolist() if r.boxes is not None else []
            polys = r.masks.xy if r.masks is not None else []
            for j, poly in enumerate(polys):
                if confs and confs[j] < args.conf:
                    continue
                if poly is None or len(poly) < 3: 
                    continue
                pts = np.round(np.array(poly)).astype(np.int32)
                pts[:, 0] += x
                pts[:, 1] += y
                # clip to page bounds
                pts[:, 0] = np.clip(pts[:, 0], 0, W - 1)
                pts[:, 1] = np.clip(pts[:, 1], 0, H - 1)
                cv2.fillPoly(page_mask, [pts], 255)

        if args.close_px > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*args.close_px+1, 2*args.close_px+1))
            page_mask = cv2.morphologyEx(page_mask, cv2.MORPH_CLOSE, k, iterations=1)

        mask_path = out_dir / f"{page_id}_mask.png"
        overlay_path = out_dir / f"{page_id}_overlay.png"
        cv2.imwrite(str(mask_path), page_mask)

        color = np.zeros_like(bgr)
        color[page_mask > 0] = (255, 0, 0)  # blue highlight
        overlay = cv2.addWeighted(bgr, 0.75, color, 0.25, 0.0)
        cv2.imwrite(str(overlay_path), overlay)

        print(f"[saved] {overlay_path}  (W={W}, H={H}, tiles={len(tile_positions(W,H,args.tile,args.overlap))})")

    print(f"Done. Outputs in: {out_dir}")

if __name__ == "__main__":
    main()
