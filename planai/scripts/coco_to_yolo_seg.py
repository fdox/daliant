import argparse, json
from pathlib import Path
from collections import defaultdict
from PIL import Image

def main():
    ap = argparse.ArgumentParser(description="Convert Label Studio COCO (Polygon) → YOLO-seg")
    ap.add_argument("--coco", required=True, help="Path to COCO JSON export (Completed only).")
    ap.add_argument("--images_root", default="datasets/wire/raw_tiles/images", help="Tiles root.")
    ap.add_argument("--out_labels_root", default="datasets/wire/labels_temp", help="Output YOLO labels dir.")
    ap.add_argument("--category", default="wire", help="Category name to keep.")
    args = ap.parse_args()

    coco_path = Path(args.coco)
    images_root = Path(args.images_root)
    out_root = Path(args.out_labels_root)
    out_root.mkdir(parents=True, exist_ok=True)

    data = json.load(open(coco_path, "r"))

    cat_id = None
    for c in data.get("categories", []):
        if c.get("name", "").lower() == args.category.lower():
            cat_id = c["id"]
            break
    if cat_id is None:
        raise SystemExit(f"Category '{args.category}' not found in COCO categories.")

    imgs = {im["id"]: im for im in data.get("images", [])}
    annos_by_img = defaultdict(list)
    for ann in data.get("annotations", []):
        annos_by_img[ann["image_id"]].append(ann)

    n_imgs = 0
    n_polys = 0
    n_empty = 0

    for img_id, im in imgs.items():
        file_name = Path(im["file_name"]).name
        stem = Path(file_name).stem

        w, h = im.get("width"), im.get("height")
        if not w or not h:
            with Image.open(images_root / file_name) as pil_im:
                w, h = pil_im.size

        lines = []
        for ann in annos_by_img.get(img_id, []):
            if ann.get("category_id") != cat_id:
                continue
            seg = ann.get("segmentation", [])
            if isinstance(seg, list):
                for flat in seg:
                    if not isinstance(flat, list) or len(flat) < 6:
                        continue
                    coords = []
                    for i in range(0, len(flat), 2):
                        x = max(0.0, min(float(flat[i]),   float(w))) / float(w)
                        y = max(0.0, min(float(flat[i+1]), float(h))) / float(h)
                        coords.append(f"{x:.6f} {y:.6f}")
                    lines.append("0 " + " ".join(coords))
                    n_polys += 1
            elif isinstance(seg, dict):
                print(f"[info] RLE mask found for {file_name}; skipping (expected polygons).")

        out_txt = out_root / f"{stem}.txt"
        with open(out_txt, "w") as f:
            f.write("\n".join(lines))

        n_imgs += 1
        if not lines:
            n_empty += 1

    print(f"Converted {n_imgs} images → {out_root}")
    print(f"Polygons written: {n_polys} | Empty label files (true negatives): {n_empty}")

if __name__ == "__main__":
    main()
