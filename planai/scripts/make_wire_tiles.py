import argparse, csv, sys
from pathlib import Path
from PIL import Image, ImageOps

def try_import_pdf2image():
    try:
        from pdf2image import convert_from_path
        return convert_from_path
    except Exception:
        print("ERROR: pdf2image not available. Install with `pip install pdf2image pillow` and ensure Poppler is installed.", file=sys.stderr)
        sys.exit(1)

def pad_to_tile(img, tile):
    w, h = img.size
    pw, ph = max(0, tile - w), max(0, tile - h)
    return ImageOps.expand(img, border=(0, 0, pw, ph), fill=255) if (pw or ph) else img

def tile_image(img, tile=1536, overlap=0.30):
    w, h = img.size
    stride = max(1, int(tile * (1 - overlap)))
    xs = list(range(0, max(1, w - tile + 1), stride))
    ys = list(range(0, max(1, h - tile + 1), stride))
    if xs[-1] != max(0, w - tile): xs.append(max(0, w - tile))
    if ys[-1] != max(0, h - tile): ys.append(max(0, h - tile))
    for y in ys:
        for x in xs:
            crop = img.crop((x, y, x + tile, y + tile))
            yield x, y, pad_to_tile(crop, tile)

def main():
    ap = argparse.ArgumentParser(description="Tile a PDF (or page images) into fixed-size tiles + per-run map.csv")
    ap.add_argument("--inp", required=True, help="Path to input PDF or a folder of page images (png/jpg).")
    ap.add_argument("--out", required=True, help="Output folder for tiles (will be created).")
    ap.add_argument("--tile", type=int, default=1536)
    ap.add_argument("--overlap", type=float, default=0.30)
    ap.add_argument("--dpi", type=int, default=300, help="DPI for rendering PDF pages.")
    ap.add_argument("--fmt", choices=["png","jpg","jpeg"], default="png")
    ap.add_argument("--max_pages", type=int, default=0, help="0 = all pages")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    map_csv = out_dir / "map.csv"

    src = Path(args.inp)

    pages = []
    if src.is_file() and src.suffix.lower() == ".pdf":
        convert_from_path = try_import_pdf2image()
        pil_pages = convert_from_path(str(src), dpi=args.dpi)
        if args.max_pages:
            pil_pages = pil_pages[:args.max_pages]
        for i, p in enumerate(pil_pages, start=1):
            pages.append((f"{src.stem}_p{i:03d}", p.convert("RGB")))
    elif src.is_dir():
        imgs = sorted([p for p in src.iterdir() if p.suffix.lower() in {".png",".jpg",".jpeg"}])
        if args.max_pages:
            imgs = imgs[:args.max_pages]
        for i, p in enumerate(imgs, start=1):
            pages.append((f"{p.stem}", Image.open(p).convert("RGB")))
    else:
        print("Input must be a PDF file or a folder of page images.", file=sys.stderr)
        sys.exit(1)

    rows = []
    n_tiles = 0
    for pid, page_img in pages:
        for x, y, crop in tile_image(page_img, tile=args.tile, overlap=args.overlap):
            fname = f"{pid}_x{x}_y{y}_s{args.tile}.{args.fmt}"
            save_path = out_dir / fname
            if args.fmt in ("jpg","jpeg"):
                crop.save(save_path, quality=95)
            else:
                crop.save(save_path)
            rows.append([fname, pid, x, y, args.tile])
            n_tiles += 1

    with open(map_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tile_file","page_id","x","y","tile"])
        w.writerows(rows)

    print(f"Wrote {n_tiles} tiles to {out_dir}")
    print(f"Per-run mapping saved to {map_csv}")

if __name__ == "__main__":
    main()
