"""Figures + numbers for the camera 5-fold CV run (report chapter 4/5).

Reads experiments/kaggle_cv_run/cv_out (outputs of cv_kfold_kaggle.ipynb on Kaggle) and writes
report/figures/cv_*.png and report/cv_results.json.   Usage (from midterm/): python experiments/cv_analysis.py
"""
import contextlib
import glob
import io
import json
import os
import shutil
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

CLASSES = ['Car', 'Motorcycle', 'Bus', 'Truck', 'Tuktuk', 'Van', 'Pickup', 'Songthaew']
CV = Path('experiments/kaggle_cv_run/cv_out')
FIG = Path('report/figures')
RAW = Path('data/raw')
gt = pd.read_csv(RAW / 'train.csv')
files = sorted(os.listdir(RAW / 'train' / 'train'))
cam = lambda f: f.split('_')[0]


def coco_map50(pred, fs):
    fid = {f: i for i, f in enumerate(fs)}
    g = gt[gt.image_id.isin(fid)]
    p = pred[pred.image_id.isin(fid)]
    ds = {'images': [{'id': i, 'width': 352, 'height': 288} for i in fid.values()],
          'categories': [{'id': k} for k in range(8)],
          'annotations': [{'id': k + 1, 'image_id': fid[r.image_id], 'category_id': int(r.class_id),
                           'bbox': [r.x1, r.y1, r.x2 - r.x1, r.y2 - r.y1], 'area': (r.x2 - r.x1) * (r.y2 - r.y1),
                           'iscrowd': 0} for k, r in enumerate(g.itertuples())]}
    dets = [{'image_id': fid[r.image_id], 'category_id': int(r.class_id), 'score': r.confidence,
             'bbox': [r.x1, r.y1, r.x2 - r.x1, r.y2 - r.y1]} for r in p.itertuples()]
    with contextlib.redirect_stdout(io.StringIO()):
        cg = COCO(); cg.dataset = ds; cg.createIndex()
        E = COCOeval(cg, cg.loadRes(dets), 'bbox'); E.evaluate(); E.accumulate(); E.summarize()
    pr = E.eval['precision'][0, :, :, 0, 2]
    per = {CLASSES[k]: (round(float(pr[:, k][pr[:, k] > -1].mean()), 4) if (pr[:, k] > -1).any() else None) for k in range(8)}
    return round(float(E.stats[1]), 4), per


folds = [json.load(open(p)) for p in sorted(glob.glob(str(CV / 'fold*' / 'metrics.json')))]
oof = pd.concat([pd.read_csv(CV / f'fold{m["fold"]}' / 'val_pred.csv') for m in folds])
oof_map, oof_ap = coco_map50(oof, files)
old4 = [f for f in files if cam(f) in ['1066', '1427', '1437', '244']]
m4, ap4 = coco_map50(oof, old4)
s4 = json.load(open('submissions/yolo26s_960_tta.json'))       # YOLO26s trained without these 4 cameras
epochs = {}
for m in folds:
    r = pd.read_csv(CV / f'fold{m["fold"]}' / 'results.csv'); r.columns = r.columns.str.strip()
    epochs[m['fold']] = {'epochs_run': int(len(r)), 'best_epoch': int(r.epoch[r['metrics/mAP50(B)'].idxmax()])}
res = {'folds': [{**m, **epochs[m['fold']]} for m in folds], 'oof_map50': oof_map, 'oof_ap': oof_ap,
       'old4_m': m4, 'old4_m_ap': ap4, 'old4_s': s4['val_map50'], 'old4_s_ap': s4['val_ap']}
json.dump(res, open('report/cv_results.json', 'w'), indent=2)
print(json.dumps({k: v for k, v in res.items() if k != 'folds'}, indent=1))

# 1) validation mAP50 per epoch for every fold
fig, ax = plt.subplots(figsize=(9, 3.8))
for m in folds:
    r = pd.read_csv(CV / f'fold{m["fold"]}' / 'results.csv'); r.columns = r.columns.str.strip()
    ax.plot(r.epoch, r['metrics/mAP50(B)'], label=f"fold {m['fold']} ({', '.join(m['val_cams'])})")
ax.set_xlabel('epoch'); ax.set_ylabel('val mAP50 (Ultralytics)'); ax.set_title('YOLO26m 960 - validation mAP50 per fold'); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(FIG / 'cv_fold_curves.png', dpi=110); plt.close()

# 2) per-class AP: YOLO26s vs YOLO26m on the same 4 unseen cameras, and OOF over all 15 cameras
x = np.arange(8); w = 0.27
fig, ax = plt.subplots(figsize=(10, 3.8))
for i, (vals, lab) in enumerate([(s4['val_ap'], f"YOLO26s, 4 cams ({s4['val_map50']:.3f})"),
                                 (ap4, f'YOLO26m CV, same 4 cams ({m4:.3f})'),
                                 (oof_ap, f'YOLO26m CV, all 15 cams ({oof_map:.3f})')]):
    b = ax.bar(x + (i - 1) * w, [vals[c] or 0 for c in CLASSES], w, label=lab)
ax.set_xticks(x); ax.set_xticklabels(CLASSES); ax.set_ylim(0, 1); ax.set_ylabel('AP@50'); ax.legend(fontsize=8)
ax.set_title('Per-class AP@50 on cameras unseen during training')
plt.tight_layout(); plt.savefig(FIG / 'cv_per_class.png', dpi=110); plt.close()

# 3) ensemble predictions on test cameras
PALETTE = [(255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29), (207, 210, 49), (72, 249, 10), (26, 147, 52), (0, 212, 187)]
ens = pd.read_csv('submissions/yolo26m_960_cv5fold_wbf.csv')
test_dir = RAW / 'test' / 'test'
tfiles = sorted(os.listdir(test_dir)); rng = np.random.default_rng(42)
pick = [f for c in sorted({cam(f) for f in tfiles}) for f in rng.choice([t for t in tfiles if cam(t) == c], 2, replace=False)]
fig, axes = plt.subplots(2, 5, figsize=(19, 6.4))
for ax, f in zip(axes.T.ravel(), pick):
    im = cv2.resize(cv2.cvtColor(cv2.imread(str(test_dir / f)), cv2.COLOR_BGR2RGB), None, fx=2, fy=2)
    for r in ens[(ens.image_id == f) & (ens.confidence > 0.3)].itertuples():
        c = PALETTE[r.class_id]
        cv2.rectangle(im, (int(r.x1 * 2), int(r.y1 * 2)), (int(r.x2 * 2), int(r.y2 * 2)), c, 2)
        cv2.putText(im, CLASSES[r.class_id], (int(r.x1 * 2), max(int(r.y1 * 2) - 3, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1)
    ax.imshow(im); ax.set_title(f, fontsize=8); ax.axis('off')
plt.tight_layout(); plt.savefig(FIG / 'cv_ensemble_test_pred.png', dpi=100); plt.close()

shutil.copy(CV / 'fold1' / 'confusion_matrix_normalized.png', FIG / 'cv_fold1_confusion.png')
print('figures written')
