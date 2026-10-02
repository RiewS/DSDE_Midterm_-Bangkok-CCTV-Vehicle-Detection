"""Fuse per-fold test predictions with Weighted Boxes Fusion (numba-free import).

Usage (from midterm/):
  python experiments/ensemble_wbf.py OUT.csv IOU CONF_TYPE pred_dir1 [pred_dir2 ...]
Each pred_dir contains fold*/test_pred.csv (outputs of cv_kfold_kaggle*.ipynb).
"""
import glob, importlib.util, os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sp = importlib.util.find_spec('ensemble_boxes')
spec = importlib.util.spec_from_file_location('ensemble_boxes_wbf', os.path.join(os.path.dirname(sp.origin), 'ensemble_boxes_wbf.py'))
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
W, H = 352, 288
out, iou, ctype, dirs = sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4:]
paths = sorted(p for d in dirs for p in glob.glob(os.path.join(d, 'fold*', 'test_pred.csv')))
print('models:', len(paths)); [print(' ', p) for p in paths]
preds = [dict(tuple(pd.read_csv(p).groupby('image_id'))) for p in paths]
test_files = sorted(os.listdir('data/raw/test/test'))
rows = []
for f in test_files:
    bl, sl, ll = [], [], []
    for d in preds:
        g = d.get(f)
        if g is None:
            bl.append(np.zeros((0, 4))); sl.append(np.zeros(0)); ll.append(np.zeros(0)); continue
        bl.append(np.clip(g[['x1', 'y1', 'x2', 'y2']].values / [W, H, W, H], 0, 1)); sl.append(g.confidence.values); ll.append(g.class_id.values)
    if not any(len(s) for s in sl):
        continue
    b, s, l = mod.weighted_boxes_fusion(bl, sl, ll, iou_thr=iou, skip_box_thr=0.001, conf_type=ctype)
    for (x1, y1, x2, y2), sc, lb in zip(b * [W, H, W, H], s, l):
        rows.append((f, int(lb), float(sc), x1, y1, x2, y2))
ens = pd.DataFrame(rows, columns=['image_id', 'class_id', 'confidence', 'x1', 'y1', 'x2', 'y2'])
ens = ens.round({'confidence': 5, 'x1': 2, 'y1': 2, 'x2': 2, 'y2': 2})
ens = ens[(ens.x2 > ens.x1) & (ens.y2 > ens.y1)]
ens = ens.sort_values(['image_id', 'confidence'], ascending=[True, False]).groupby('image_id').head(300).reset_index(drop=True)
ens.insert(0, 'id', range(len(ens)))
ens.to_csv(out, index=False)
print('saved', out, len(ens), 'rows,', ens.image_id.nunique(), 'images')
