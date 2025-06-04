#!/usr/bin/env python
"""
Verify that the custom WaveBranch is (1) in the model graph and
(2) participates in training (i.e. its parameters get non-zero grads).

Usage
-----
$ python scripts/verify_wave_branch.py
"""

import torch
import sys
from pathlib import Path

#### ------------------------------------------------------------------ ####
#### 1.  Find repo root and make sure the customised YOLOv5 is on sys.path
#### ------------------------------------------------------------------ ####
root = Path(__file__).resolve().parents[1]          # kv260-tiny-det/
yolo_dir = root / 'training' / 'wtdbb' / 'yolov5'   # customised fork
if not yolo_dir.exists():
    sys.exit(f"❌  Expected repo at {yolo_dir} – not found.")

sys.path.insert(0, str(yolo_dir))

#### ------------------------------------------------------------------ ####
#### 2.  Build the model (WaveBranch is injected in Model.__init__)
#### ------------------------------------------------------------------ ####
from models.yolo import Model                      # noqa:  E402

model_cfg = yolo_dir / 'models' / 'yolov5s.yaml'   # any YOLO yaml is fine
model = Model(str(model_cfg))
model.train()                                      # training mode = WaveBranch active

#### ------------------------------------------------------------------ ####
#### 3.  Forward + dummy loss + backward
#### ------------------------------------------------------------------ ####
dummy = torch.randn(1, 3, 640, 640)                # fake image batch
pred  = model(dummy)[0]                            # forward; tuple -> tensor
loss  = pred.sum()                                 # Scalar fake loss
loss.backward()                                    # populate .grad fields

#### ------------------------------------------------------------------ ####
#### 4.  Collect WaveBranch information
#### ------------------------------------------------------------------ ####
wave_modules = [m for m in model.modules()
                if m.__class__.__name__ == 'WaveBranch']

if not wave_modules:
    sys.exit("❌  WaveBranch NOT found in the model graph!")

grad_sum = sum(p.grad.abs().sum().item()
               for m in wave_modules
               for p in m.parameters()
               if p.grad is not None)

param_cnt = sum(p.numel() for m in wave_modules for p in m.parameters())

#### ------------------------------------------------------------------ ####
#### 5.  Report
#### ------------------------------------------------------------------ ####
print("\n✔ WaveBranch present in graph")
print(f"   parameters : {param_cnt:,}")
print(f"   Σ|grad|     : {grad_sum:.6f}")

if grad_sum == 0:
    print("⚠  WaveBranch got ZERO gradient – something is wrong.")
else:
    print("👍 Non-zero gradient → WaveBranch participates in training.")
