#!/usr/bin/env python
import argparse, pathlib, yaml, sys, torch, numpy as np
from tqdm import tqdm

# ---------- repo paths (ONLY WT-DBB) -------------
proj_root = pathlib.Path(__file__).resolve().parents[1]
wt_repo   = proj_root / "training" / "wtdbb" / "yolov5"
sys.path.insert(0, str(wt_repo.resolve()))

from utils.dataloaders import LoadImagesAndLabels
from utils.general     import non_max_suppression, scale_coords
from utils.metrics     import ap_per_class
from models.experimental import attempt_load

# ---------- helpers ------------------------------
def val_list_from_yaml(yaml_path: pathlib.Path):
    data = yaml.safe_load(yaml_path.read_text())
    root = yaml_path.parent / data.get("path", "")
    root = root.resolve()
    val  = pathlib.Path(data["val"])
    return (root / val).resolve() if not val.is_absolute() else val

def ap_tiny(weights: pathlib.Path, val_path: pathlib.Path, imgsz=640):
    model  = attempt_load(weights, device="cpu")
    model.eval()
    stride = int(model.stride.max())
    loader = LoadImagesAndLabels(
        path=str(val_path),
        img_size=imgsz,
        batch_size=1,
        stride=stride,
        rect=True,
        pad=0.5,
        prefix='',
        hyp={}             # <-- add this line
    )

    stats = []
    for imgs, targets, *_ in tqdm(loader, desc=weights.stem):
        imgs = imgs.float() / 255.0
        with torch.no_grad():
            preds = model(imgs)[0]
        preds = non_max_suppression(preds, 0.001, 0.6)[0]
        if preds is None: continue
        preds[:, :4] = scale_coords(imgs.shape[2:], preds[:, :4],
                                    loader.imgs[0].shape).round()
        area = (preds[:,2]-preds[:,0]) * (preds[:,3]-preds[:,1])
        preds = preds[area < 256]
        if len(preds)==0: continue
        stats.append((preds[:,4]>0.5, preds[:,4], preds[:,5], targets[:,0]))
    if not stats: return 0.0
    stats = [np.concatenate(x,0) for x in zip(*stats)]
    _,_,ap,_,_ = ap_per_class(*stats)
    return float(ap.mean())

# ---------- CLI ----------------------------------
p = argparse.ArgumentParser()
p.add_argument("--weights", type=pathlib.Path, required=True)
p.add_argument("--yaml",    type=pathlib.Path, required=True)
p.add_argument("--imgsz",   type=int, default=640)
args = p.parse_args()

val_path = val_list_from_yaml(args.yaml.resolve())
score    = ap_tiny(args.weights.resolve(), val_path, args.imgsz)
print(f"AP<16 px = {score:.3f}")
