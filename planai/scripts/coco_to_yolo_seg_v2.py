import argparse, json, re
from pathlib import Path
from collections import defaultdict
from typing import Optional, Tuple

try:
    from PIL import Image
    PIL_OK = True
except Exception:
    PIL_OK = False

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

tile_re = re.compile(
    r"""(?xi)
    ^(?:[a-z0-9]{6,}-)?        # optional LS hash prefix + '-'
    (?P<doc>.+?)               # document base (greedy, up to _p)
    _p(?P<p>\d+)
    _x(?P<x>\d+)
    _y(?P<y>\d+)
    (?:_s(?P<s>\d+))?          # optional tile size suffix
    $
    """
)

def parse_tokens(stem: str) -> Optional[Tuple[str,int,int,int,Optional[int]]]:
    m = tile_re.match(stem)
    if not m:
        return None
    doc = m.group("doc")
    p   = int(m.group("p"))
    x   = int(m.group("x"))
    y   = int(m.group("y"))
    s   = int(m.group("s")) if m.group("s") else None
    return doc, p, x, y, s

def index_local_images(images_root: Path):
    by_dpxy = {}         # (doc,p,x,y) -> Path
    by_pxy  = defaultdict(list)  # (p,x,y) -> [Path,...]
    for p in images_root.iterdir():
        if not p.is_file() or p.suffix.lower() not in IMG_EXTS:
            continue
        tokens = parse_tokens(p.stem)
        if not tokens:
            continue
        doc, pg, xx, yy, _ = tokens
        by_dpxy[(doc, pg, xx, yy)] = p
        by_pxy[(pg, xx, yy)].append(p)
    return by_dpxy, by_pxy

def main():
    ap = argparse.ArgumentParser(description="Convert Label Studio COCO → YOLO-seg (robust name mapping).")
    ap.add_argument("--coco", required=True, help="Path to COCO JSON export (Completed only).")
    ap.add_argument("--images_root", default="datasets/wire/raw_tiles/images", help="Local tiles root.")
    ap.add_argument("--out_labels_root", default="datasets/wire/labels_temp", help="Output YOLO labels dir.")
    ap.add_argument("--category", default="wire", help="Category name to keep.")
    ap.add_argument("--tile_size", type=int, default=1536, help="Fallback W/H if COCO lacks size and image not found.")
    args = ap.parse_args()

    coco_path = Path(args.coco)
    images_root = Path(args.images_root)
    out_root = Path(args.out_labels_root)
    out_root.mkdir(parents=True, exist_ok=True)

    with open(coco_path, "r") as f:
        data = json.load(f)

    cat_id = None
    for c in data.get("categories", []):
        if c.get("name", "").lower() == args.category.lower():
            cat_id = c["id"]; break
    if cat_id is None:
        raise SystemExit(f"Category '{args.category}' not found in COCO categories.")

    by_dpxy, by_pxy = index_local_images(images_root)

    imgs = {im["id"]: im for im in data.get("images", [])}
    annos_by_img = defaultdict(list)
    for ann in data.get("annotations", []):
        annos_by_img[ann["image_id"]].append(ann)

    n_imgs = 0
    n_polys = 0
    n_empty = 0
    n_mapped = 0
    n_unmapped = 0

    for img_id, im in imgs.items():
        file_name = Path(im.get("file_name", "")).name
        stem_src  = Path(file_name).stem
        toks = parse_tokens(stem_src)

        local_path = None
        if toks:
            doc,p,x,y,_ = toks
            local_path = by_dpxy.get((doc,p,x,y))
            if not local_path:
                # fallback: match by p,x,y only (safe if working on one PDF)
                cand = by_pxy.get((p,x,y), [])
                if len(cand) == 1:
                    local_path = cand[0]
                elif len(cand) > 1:
                    # choose the one whose stem contains doc (best effort)
                    for c in cand:
                        if doc in c.stem:
                            local_path = c; break
                    if not local_path:
                        local_path = cand[0]  # last resort
        if local_path:
            stem_out = local_path.stem
            n_mapped += 1
        else:
            stem_out = stem_src
            n_unmapped += 1
            print(f"[warn] No local match for {file_name}; writing label as {stem_out}.txt")

        w = im.get("width")
        h = im.get("height")
        if (not w or not h):
            if local_path and PIL_OK:
                try:
                    with Image.open(local_path) as pil_im:
                        w, h = pil_im.size
                except Exception:
                    w = h = None
        if (not w or not h):
            w = h = args.tile_size  # safe fallback for square tiles

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
                        x_f = max(0.0, min(float(flat[i]),   float(w))) / float(w)
                        y_f = max(0.0, min(float(flat[i+1]), float(h))) / float(h)
                        coords.append(f"{x_f:.6f} {y_f:.6f}")
                    lines.append("0 " + " ".join(coords))
                    n_polys += 1
            elif isinstance(seg, dict):
                pass

        out_txt = out_root / f"{stem_out}.txt"
        with open(out_txt, "w") as f:
            f.write("\n".join(lines))

        n_imgs += 1
        if not lines:
            n_empty += 1

    print(f"Converted {n_imgs} images → {out_root}")
    print(f"Mapped to local tiles: {n_mapped} | Unmapped (fallback names): {n_unmapped}")
    print(f"Polygons written: {n_polys} | Empty label files (true negatives): {n_empty}")

if __name__ == "__main__":
    main()
