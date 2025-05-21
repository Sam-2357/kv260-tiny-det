#!/usr/bin/env python3
"""
Convert SeaDronesSee COCO-style JSON to YOLO txt files.
Usage:
    python scripts/sds_to_yolo.py \
        --json  annotations_train.json \
        --imgs  /path/to/images \
        --out   data/SDS_yolo/train
"""

import json, argparse, shutil, pathlib
from tqdm import tqdm

ap = argparse.ArgumentParser()
ap.add_argument("--json", required=True, help="COCO annotation file")
ap.add_argument("--imgs", required=True, help="folder with original images")
ap.add_argument("--out",  required=True, help="output dir (YOLO layout)")
args = ap.parse_args()

jpath   = pathlib.Path(args.json)
imgroot = pathlib.Path(args.imgs)
outdir  = pathlib.Path(args.out)
(lbl_dir, img_dir) = (outdir / "labels", outdir / "images")
lbl_dir.mkdir(parents=True, exist_ok=True)
img_dir.mkdir(parents=True, exist_ok=True)

data = json.load(jpath.open())
imgs = {im["id"]: im for im in data["images"]}
labels = {im_id: [] for im_id in imgs}

for ann in data["annotations"]:
    if ann.get("iscrowd", 0):          # skip crowd boxes
        continue
    x, y, w, h = ann["bbox"]
    im = imgs[ann["image_id"]]
    xc, yc = x + w / 2, y + h / 2
    # SeaDronesSee class mapping: all 0 (single-class) — change if needed
    labels[im["id"]].append(
        f"0 {xc/im['width']:.6f} {yc/im['height']:.6f} "
        f"{w/im['width']:.6f} {h/im['height']:.6f}"
    )

for im_id, rows in tqdm(labels.items(), desc="writing"):
    stem = pathlib.Path(imgs[im_id]["file_name"]).stem
    (lbl_dir / f"{stem}.txt").write_text("\n".join(rows))
    src = imgroot / imgs[im_id]["file_name"]
    dst = img_dir  / imgs[im_id]["file_name"]
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copy(src, dst)
print("✓ conversion done →", outdir)

