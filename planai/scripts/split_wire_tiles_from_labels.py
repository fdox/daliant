import argparse, shutil, random
from pathlib import Path
import re
from collections import defaultdict

IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

tile_re = re.compile(
    r"""(?xi)
    ^(?:[a-z0-9]{6,}-)?        # optional hashed prefix from LS
    (?P<doc>.+?)               # document base name
    _p(?P<p>\d+)
    _x(?P<x>\d+)
    _y(?P<y>\d+)
    (?:_s(?P<s>\d+))?          # optional tile size
    $
    """
)

def parse_tokens(stem: str):
    m = tile_re.match(stem)
    if not m:
        return None
    return {
        "doc": m.group("doc"),
        "page": int(m.group("p")),
        "x": int(m.group("x")),
        "y": int(m.group("y")),
        "s": int(m.group("s")) if m.group("s") else None,
    }

def find_image(images_root: Path, stem: str) -> Path | None:
    for ext in IMG_EXTS:
        p = images_root / f"{stem}{ext}"
        if p.exists():
            return p
    return None

def write_manifest(paths, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(str(p) for p in sorted(paths)) + "\n")

def main():
    ap = argparse.ArgumentParser(description="Split YOLO-seg tiles into train/val from label files only.")
    ap.add_argument("--labels_root", default="datasets/wire/labels_temp")
    ap.add_argument("--images_root", default="datasets/wire/raw_tiles/images")
    ap.add_argument("--out_root",    default="datasets/wire")
    ap.add_argument("--val_pages", type=int, default=1, help="Hold out this many full pages if >=2 pages are present.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_ratio_if_single_page", type=float, default=0.25, help="Tile-level split if only one page.")
    args = ap.parse_args()

    random.seed(args.seed)
    labels_root = Path(args.labels_root)
    images_root = Path(args.images_root)
    out_root    = Path(args.out_root)

    label_files = sorted(labels_root.glob("*.txt"))
    if not label_files:
        raise SystemExit(f"No label files found in {labels_root}")

    records = []
    by_page = defaultdict(list)
    for lbl in label_files:
        stem = lbl.stem
        toks = parse_tokens(stem)
        if not toks:
            print(f"[warn] Could not parse page from: {stem} -> skipping")
            continue
        img_path = find_image(images_root, stem)
        if not img_path:
            print(f"[warn] No image found for: {stem} -> skipping")
            continue
        rec = {"stem": stem, "page": toks["page"], "img": img_path, "lbl": lbl}
        records.append(rec)
        by_page[toks["page"]].append(rec)

    if not records:
        raise SystemExit("No usable (image,label) pairs found.")

    train_img = out_root / "train" / "images"
    train_lbl = out_root / "train" / "labels"
    val_img   = out_root / "val"   / "images"
    val_lbl   = out_root / "val"   / "labels"
    for d in [train_img, train_lbl, val_img, val_lbl]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    uniq_pages = sorted(by_page.keys())
    train_set, val_set = [], []

    if len(uniq_pages) >= 2:
        val_pg = uniq_pages[-args.val_pages:] if args.val_pages < len(uniq_pages) else uniq_pages[-1:]
        train_pg = [p for p in uniq_pages if p not in val_pg]
        for p in train_pg:
            train_set.extend(by_page[p])
        for p in val_pg:
            val_set.extend(by_page[p])
        print(f"Held-out pages for val: {val_pg} | train pages: {train_pg}")
    else:
        k = int(round(len(records) * args.val_ratio_if_single_page))
        k = max(1, min(len(records)-1, k))
        recs = records[:]
        random.shuffle(recs)
        val_set = recs[:k]
        train_set = recs[k:]
        print(f"Single-page dataset → tile-level split: val tiles={len(val_set)}, train tiles={len(train_set)}")

    train_img_paths, val_img_paths = [], []
    for rec in train_set:
        shutil.copy2(rec["img"], train_img / rec["img"].name)
        shutil.copy2(rec["lbl"], train_lbl / rec["lbl"].name)
        train_img_paths.append(train_img / rec["img"].name)
    for rec in val_set:
        shutil.copy2(rec["img"], val_img / rec["img"].name)
        shutil.copy2(rec["lbl"], val_lbl / rec["lbl"].name)
        val_img_paths.append(val_img / rec["img"].name)

    man_root = Path("manifests"); man_root.mkdir(exist_ok=True)
    write_manifest(train_img_paths, man_root / "wire_train.txt")
    write_manifest(val_img_paths,   man_root / "wire_val.txt")

    tn_train = sum(1 for r in train_set if Path(train_lbl / f"{r['stem']}.txt").stat().st_size == 0)
    tn_val   = sum(1 for r in val_set   if Path(val_lbl   / f"{r['stem']}.txt").stat().st_size == 0)
    print(f"Train: {len(train_set)} tiles (true negatives: {tn_train})")
    print(f"Val:   {len(val_set)} tiles (true negatives: {tn_val})")
    print(f"Wrote manifests: manifests/wire_train.txt and manifests/wire_val.txt")

if __name__ == "__main__":
    main()
