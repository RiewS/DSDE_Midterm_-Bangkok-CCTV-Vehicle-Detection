"""Post-processing study on the validation cameras (no training).

Evaluates a trained model with different MAX_DET values, with/without TTA, and reports pycocotools mAP@50
exactly like the notebook. Usage (from midterm/):
    python experiments/postproc_eval.py runs/yolo26s_640/weights/best.pt 640
"""
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO

CLASSES = ['Car', 'Motorcycle', 'Bus', 'Truck', 'Tuktuk', 'Van', 'Pickup', 'Songthaew']
VAL_CAMS = ['1066', '1427', '1437', '244']
W, H = 352, 288
weights, imgsz = sys.argv[1], int(sys.argv[2])
img_dir = Path('data/raw/train/train')
gt = pd.read_csv('data/raw/train.csv')
files = sorted(p.name for p in img_dir.glob('*.jpg') if p.name.split('_')[0] in VAL_CAMS)
fid = {f: i for i, f in enumerate(files)}
gt = gt[gt.image_id.isin(fid)]
ds = {'images': [{'id': i, 'file_name': f, 'width': W, 'height': H} for f, i in fid.items()],
      'categories': [{'id': k, 'name': n} for k, n in enumerate(CLASSES)],
      'annotations': [{'id': k + 1, 'image_id': fid[r.image_id], 'category_id': int(r.class_id),
                       'bbox': [r.x1, r.y1, r.x2 - r.x1, r.y2 - r.y1], 'area': (r.x2 - r.x1) * (r.y2 - r.y1), 'iscrowd': 0}
                      for k, r in enumerate(gt.itertuples())]}


def evaluate(dets):
    with contextlib.redirect_stdout(io.StringIO()):
        cg = COCO(); cg.dataset = ds; cg.createIndex()
        E = COCOeval(cg, cg.loadRes(dets), 'bbox'); E.evaluate(); E.accumulate(); E.summarize()
    prec = E.eval['precision'][0, :, :, 0, 2]
    return E.stats[1], {CLASSES[k]: round(float(prec[:, k][prec[:, k] > -1].mean()), 4) for k in range(8)}


model = YOLO(weights)
out = {}
for tta in (False, True):
    rows = []
    paths = [str(img_dir / f) for f in files]
    for i in range(0, len(paths), 32):
        for p, r in zip(paths[i:i + 32], model.predict(paths[i:i + 32], imgsz=imgsz, conf=0.001, max_det=300,
                                                         augment=tta, verbose=False)):
            b = r.boxes
            for (x1, y1, x2, y2), c, s in zip(b.xyxy.cpu().numpy(), b.cls.cpu().numpy(), b.conf.cpu().numpy()):
                rows.append((fid[Path(p).name], int(c), float(s), float(x1), float(y1), float(x2 - x1), float(y2 - y1)))
    df = pd.DataFrame(rows, columns=['image_id', 'category_id', 'score', 'x', 'y', 'w', 'h'])
    for md in (100, 300):
        d = df.sort_values('score', ascending=False).groupby('image_id').head(md)
        dets = [{'image_id': int(r.image_id), 'category_id': int(r.category_id), 'score': r.score,
                 'bbox': [r.x, r.y, r.w, r.h]} for r in d.itertuples()]
        m, per = evaluate(dets)
        key = f"tta={tta} max_det={md}"
        out[key] = {'map50': round(float(m), 4), 'per_class': per}
        print(key, round(m, 4), per, flush=True)
Path(weights).with_name('postproc_eval.json').write_text(json.dumps(out, indent=2))
