#!/usr/bin/env python
"""
Compute mAP for very small objects (<16×16 px) on the validation split.

Usage:
    python scripts/eval_tiny_ap.py \
        --weights runs/train/wtdbb_s/weights/best.pt \
        --yaml    data/seadronessee.yaml \
        --repo_root . \
        --imgsz 640
"""

# 0. standard libs
import argparse, pathlib, sys, yaml, numpy as np, torch
from tqdm import tqdm

# 1. add both YOLOv5 repos to sys.path
def add_yolo_paths(project_root: pathlib.Path):
    wt_path  = project_root / 'training' / 'wtdbb' / 'yolov5'
    base_path = project_root.parent / 'yolov5_baseline'
    sys.path.insert(0, str(wt_path.resolve()))
    sys.path.insert(0, str(base_path.resolve()))

# 2. helper: load val image list from YAML
def resolve_val_list(yaml_file: pathlib.Path) -> pathlib.Path:
    data = yaml.safe_load(yaml_file.read_text())

    # ➊ dataset root (optional 'path' key)
    root_dir = yaml_file.parent
    if 'path' in data and data['path']:
        root_dir = (yaml_file.parent / data['path']).resolve()

    # ➋ val entry
    val_entry = pathlib.Path(data['val'])
    if not val_entry.is_absolute():
        val_entry = (root_dir / val_entry).resolve()

    return val_entry

# 3. evaluation core
def ap_tiny(weights: pathlib.Path, val_list: pathlib.Path, imgsz: int):
    from utils.dataloaders import LoadImagesAndLabels
    from utils.general     import non_max_suppression, scale_coords
    from utils.metrics     import ap_per_class
    from models.experimental import attempt_load

    model  = attempt_load(weights, device='cpu')
    model.eval()
    stride = int(model.stride.max())

    loader = LoadImagesAndLabels(
        path=str(val_list),
        img_size=imgsz,
        batch_size=1,
        stride=stride,
        rect=True,
        pad=0.5,
        prefix='')

    stats = []
    for imgs, targets, _, _ in tqdm(loader, desc=weights.stem):
        imgs = imgs.float() / 255.0
        with torch.no_grad():
            preds = model(imgs)[0]
        preds = non_max_suppression(preds, 0.001, 0.6)[0]
        if preds is None:
            continue
        preds[:, :4] = scale_coords(imgs.shape[2:], preds[:, :4],
                                    loader.imgs[0].shape).round()
        area = (preds[:,2]-preds[:,0]) * (preds[:,3]-preds[:,1])
        preds = preds[area < 256]          # <16 px
        if len(preds) == 0:
            continue
        stats.append((preds[:,4] > 0.5, preds[:,4], preds[:,5], targets[:,0]))

    if not stats:
        return 0.0
    stats = [np.concatenate(x, 0) for x in zip(*stats)]
    _, _, ap, _, _ = ap_per_class(*stats)
    return float(ap.mean())

# 4. arg-parser and main
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', required=True, type=pathlib.Path)
    p.add_argument('--yaml',    required=True, type=pathlib.Path)
    p.add_argument('--repo_root', default='.', type=pathlib.Path,
                   help='kv260-tiny-det project root')
    p.add_argument('--imgsz', type=int, default=640)
    args = p.parse_args()

    add_yolo_paths(args.repo_root.resolve())
    val_list = resolve_val_list(args.yaml.resolve())

    ap16 = ap_tiny(args.weights, val_list, args.imgsz)
    print(f'AP<16 px for {args.weights}: {ap16:.3f}')

if __name__ == '__main__':
    main()
