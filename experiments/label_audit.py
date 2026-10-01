"""Label audit of train.csv (no training needed).

1) Rule-based outliers as in the course Lab5 (z-score > 3 and 1.5 x IQR), applied per class to log(aspect ratio)
   and log(area), plus 'sliver' boxes (a side <= 2 px).
2) Model-assisted audit with the out-of-fold (OOF) predictions of the YOLO26m camera 5-fold CV: every image is
   predicted by a model that never saw its camera, so strong disagreements between model and label point to
   label errors:
     - possible MISSING label : prediction conf >= 0.7 that overlaps no ground-truth box (IoU < 0.3, any class)
     - possible WRONG class   : ground-truth box whose best-matching prediction (IoU >= 0.7) has another class, conf >= 0.6
     - possible BAD box       : ground-truth box with no prediction at all (IoU >= 0.3, conf >= 0.05)
Writes report/figures/audit_*.png and report/audit_results.json.   Usage (from midterm/): python experiments/label_audit.py
"""
import glob
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CLASSES = ['Car', 'Motorcycle', 'Bus', 'Truck', 'Tuktuk', 'Van', 'Pickup', 'Songthaew']
RAW = Path('data/raw'); IMG = RAW / 'train' / 'train'; FIG = Path('report/figures')
gt = pd.read_csv(RAW / 'train.csv')
gt['w'] = gt.x2 - gt.x1; gt['h'] = gt.y2 - gt.y1
gt['log_aspect'] = np.log(gt.h / gt.w); gt['log_area'] = np.log(gt.w * gt.h)
oof = pd.concat([pd.read_csv(p) for p in sorted(glob.glob('experiments/kaggle_cv_run/cv_out/fold*/val_pred.csv'))])
res = {}

# ---------------------------------------------------------------- 1) rule-based (Lab5 style)
def iqr_flag(s):
    q1, q3 = s.quantile([.25, .75]); i = q3 - q1
    return (s < q1 - 1.5 * i) | (s > q3 + 1.5 * i)

z = lambda s: (s - s.mean()) / s.std()
gt['z_aspect'] = gt.groupby('class_id').log_aspect.transform(z)
gt['z_area'] = gt.groupby('class_id').log_area.transform(z)
gt['iqr_aspect'] = gt.groupby('class_id').log_aspect.transform(iqr_flag)
gt['iqr_area'] = gt.groupby('class_id').log_area.transform(iqr_flag)
gt['sliver'] = (gt.w <= 2) | (gt.h <= 2)
rule = pd.DataFrame({
    'z>3 aspect': gt.groupby('class_id').z_aspect.apply(lambda s: int((s.abs() > 3).sum())),
    'z>3 area': gt.groupby('class_id').z_area.apply(lambda s: int((s.abs() > 3).sum())),
    'IQR aspect': gt.groupby('class_id').iqr_aspect.sum().astype(int),
    'IQR area': gt.groupby('class_id').iqr_area.sum().astype(int),
    'side <= 2 px': gt.groupby('class_id').sliver.sum().astype(int)})
rule.index = CLASSES
print(rule); res['rule_table'] = rule.reset_index().values.tolist()
res['rule_totals'] = {c: int(rule[c].sum()) for c in rule.columns}


def iou_matrix(a, b):
    lt = np.maximum(a[:, None, :2], b[None, :, :2]); rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    ar = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (ar(a)[:, None] + ar(b)[None] - inter + 1e-9)


# ---------------------------------------------------------------- 2) OOF model-assisted audit
G = dict(tuple(gt.groupby('image_id'))); P = dict(tuple(oof.groupby('image_id')))
missing, wrong, badbox = [], [], []
for f in sorted(set(G) | set(P)):
    g = G.get(f); p = P.get(f)
    gb = g[['x1', 'y1', 'x2', 'y2']].values.astype(float) if g is not None else np.zeros((0, 4))
    pb = p[['x1', 'y1', 'x2', 'y2']].values if p is not None else np.zeros((0, 4))
    M = iou_matrix(pb, gb) if len(pb) and len(gb) else np.zeros((len(pb), len(gb)))
    if p is not None:                                             # confident predictions on unlabelled places
        for i, r in enumerate(p.itertuples()):
            if r.confidence >= 0.7 and (M.shape[1] == 0 or M[i].max() < 0.3):
                missing.append((f, r.class_id, r.confidence, r.x1, r.y1, r.x2, r.y2))
    if g is not None:
        for j, r in enumerate(g.itertuples()):
            col = M[:, j] if len(pb) else np.zeros(0)
            if len(col) == 0 or not ((col >= 0.3) & (p.confidence.values >= 0.05)).any():
                badbox.append((f, r.class_id, r.x1, r.y1, r.x2, r.y2, r.w, r.h)); continue
            cand = np.where(col >= 0.7)[0]
            if len(cand):
                best = cand[np.argmax(p.confidence.values[cand])]
                pc, ps = int(p.class_id.values[best]), float(p.confidence.values[best])
                if pc != r.class_id and ps >= 0.6:
                    wrong.append((f, r.class_id, pc, ps, r.x1, r.y1, r.x2, r.y2))
missing = pd.DataFrame(missing, columns=['image_id', 'pred_class', 'conf', 'x1', 'y1', 'x2', 'y2']).sort_values('conf', ascending=False)
wrong = pd.DataFrame(wrong, columns=['image_id', 'gt_class', 'pred_class', 'conf', 'x1', 'y1', 'x2', 'y2']).sort_values('conf', ascending=False)
badbox = pd.DataFrame(badbox, columns=['image_id', 'gt_class', 'x1', 'y1', 'x2', 'y2', 'w', 'h'])
res['oof'] = {
    'possible_missing_labels': len(missing), 'missing_by_pred_class': missing.pred_class.map(dict(enumerate(CLASSES))).value_counts().to_dict(),
    'possible_wrong_class': len(wrong),
    'wrong_pairs (label -> model)': {f'{CLASSES[a]} -> {CLASSES[b]}': int(n) for (a, b), n in wrong.groupby(['gt_class', 'pred_class']).size().sort_values(ascending=False).head(10).items()},
    'possible_bad_boxes (no prediction)': len(badbox),
    'bad_boxes_side<=2px': int(((badbox.w <= 2) | (badbox.h <= 2)).sum()),
    'bad_boxes_median_size': [float(badbox.w.median()), float(badbox.h.median())]}
print(json.dumps(res['oof'], indent=1))
missing.to_csv('experiments/audit_missing.csv', index=False); wrong.to_csv('experiments/audit_wrong_class.csv', index=False)
badbox.to_csv('experiments/audit_bad_boxes.csv', index=False)


# ---------------------------------------------------------------- figures
def crop(f, boxes, pad=24, scale=3):
    """boxes: list of (x1,y1,x2,y2,color,label); crop around the first box"""
    im = cv2.cvtColor(cv2.imread(str(IMG / f)), cv2.COLOR_BGR2RGB)
    x1, y1, x2, y2 = boxes[0][:4]
    cx1, cy1 = int(max(x1 - pad, 0)), int(max(y1 - pad, 0)); cx2, cy2 = int(min(x2 + pad, 352)), int(min(y2 + pad, 288))
    c = cv2.resize(im[cy1:cy2, cx1:cx2], None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    for bx1, by1, bx2, by2, col, lab in boxes:
        p1 = (int((bx1 - cx1) * scale), int((by1 - cy1) * scale)); p2 = (int((bx2 - cx1) * scale), int((by2 - cy1) * scale))
        cv2.rectangle(c, p1, p2, col, 2); cv2.putText(c, lab, (p1[0], max(p1[1] - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    return c


def grid(items, name, title, ncols=6):
    nrows = int(np.ceil(len(items) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.6, nrows * 2.6))
    for ax in np.ravel(axes):
        ax.axis('off')
    for ax, (im, t) in zip(np.ravel(axes), items):
        ax.imshow(im); ax.set_title(t, fontsize=7)
    fig.suptitle(title, fontsize=10); plt.tight_layout(); plt.savefig(FIG / f'{name}.png', dpi=100); plt.close()


GREEN, RED = (0, 200, 0), (255, 40, 40)
gt_boxes = lambda f: [(g.x1, g.y1, g.x2, g.y2, GREEN, '') for g in G[f].itertuples()] if f in G else []
grid([(crop(r.image_id, [(r.x1, r.y1, r.x2, r.y2, RED, f'pred {CLASSES[r.pred_class]}')] + gt_boxes(r.image_id)),
       f'{r.image_id[:-4]}\n{CLASSES[r.pred_class]} {r.conf:.2f}')
      for r in missing.head(24).itertuples()], 'audit_missing',
     'OOF audit: confident predictions with no label (red = model, green = existing labels; top 24 by confidence)')
grid([(crop(r.image_id, [(r.x1, r.y1, r.x2, r.y2, GREEN, f'GT {CLASSES[r.gt_class]}')]),
       f'{r.image_id[:-4]}\nlabel {CLASSES[r.gt_class]} / model {CLASSES[r.pred_class]} {r.conf:.2f}')
      for r in wrong.head(24).itertuples()], 'audit_wrong_class', 'OOF audit: label class vs confident model class (top 24)')
bs = badbox.sample(min(24, len(badbox)), random_state=42)
grid([(crop(r.image_id, [(r.x1, r.y1, r.x2, r.y2, GREEN, f'GT {CLASSES[r.gt_class]}')]),
       f'{r.image_id[:-4]}\n{CLASSES[r.gt_class]} {r.w}x{r.h}px') for r in bs.itertuples()],
     'audit_bad_boxes', 'OOF audit: labelled boxes the model does not detect at all (random 24)')
ex = gt[gt.z_aspect.abs() > 3].assign(az=lambda d: d.z_aspect.abs()).sort_values('az', ascending=False).head(24)
grid([(crop(r.image_id, [(r.x1, r.y1, r.x2, r.y2, GREEN, CLASSES[r.class_id])]), f'{r.image_id[:-4]}\n{r.w}x{r.h}px z={r.z_aspect:.1f}')
      for r in ex.itertuples()], 'audit_zscore_aspect', 'Lab5-style rule: boxes with |z(log aspect)| > 3 (top 24)')
json.dump(res, open('report/audit_results.json', 'w'), indent=2)
print('figures written')
