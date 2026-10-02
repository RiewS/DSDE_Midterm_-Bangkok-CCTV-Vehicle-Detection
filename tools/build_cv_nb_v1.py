"""Builds midterm/cv_kfold_kaggle.ipynb  (usage: python build_cv_nb.py <out.ipynb> [folds e.g. 0,1,2])"""
import sys
import nbformat as nbf

FOLDS = sys.argv[2] if len(sys.argv) > 2 else '0,1,2,3,4'
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# Camera K-fold CV + ensemble — Bangkok CCTV Vehicle Detection (2110531 DSDE midterm)

Runs on a **Kaggle Notebook (GPU T4)** with the competition data attached, or on Colab.

* **Validation:** `StratifiedGroupKFold` with *camera* as the group → 5 folds × 3 held-out cameras. Every camera is used for
  validation exactly once (honest out-of-fold score for unseen cameras) and for training in the other 4 folds.
* **Model:** YOLO26m (COCO-pretrained), imgsz 960, same cleaning / repeat-factor sampling as `vehicle_detection.ipynb`.
* **Final prediction:** the 5 fold models (each with TTA) are fused with **Weighted Boxes Fusion**.
* **Folds** are fixed to the camera assignment of the submitted run (`FIXED_FOLDS`), because `StratifiedGroupKFold`
  gives a different assignment on other scikit-learn versions.
* **Reproduce without training:** put the submitted weights at `cv_out/fold{k}/best.pt`; those folds are only predicted.
* **Time guard:** a fold is only started if it can finish before the session limit, so the outputs are always saved.
  Folds that did not fit are run in a later session (set `FOLDS` below) and all fold outputs are combined at the end.
""")

code(r"""
# Offline install (Kaggle notebooks without internet): wheels + weights come from the private dataset
# 'dsde-midterm-offline-pkgs'. With internet (Colab), the normal pip install is used instead.
import glob, os, shutil, subprocess, sys, traceback
def _log_exc(shell, etype, evalue, tb, tb_offset=None):      # keep a copy of any error in the outputs (progress.log)
    with open('progress.log', 'a') as fh:
        fh.write(''.join(traceback.format_exception(etype, evalue, tb)))
    shell.showtraceback((etype, evalue, tb), tb_offset=tb_offset)
get_ipython().set_custom_exc((BaseException,), _log_exc)
def plog(msg):
    with open('progress.log', 'a') as fh:
        print(msg, file=fh)
whl = glob.glob('/kaggle/input/**/ultralytics-8.4.163-py3-none-any.whl', recursive=True)
if whl:
    PKG_DIR = os.path.dirname(whl[0])
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-index', '--no-deps'] + glob.glob(f'{PKG_DIR}/*.whl'), check=True)
    for f in ['yolo26m.pt', 'yolo26n.pt']:            # yolo26n.pt is used by the AMP check
        shutil.copy(f'{PKG_DIR}/{f}', f)
else:
    PKG_DIR = None
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'ultralytics==8.4.163', 'ensemble-boxes', 'pycocotools', 'scikit-learn'], check=True)
print('packages from', PKG_DIR or 'PyPI'); plog('install ok')
""")
code(r"""
import os, re, json, glob, shutil, time, math, random, contextlib, io
from pathlib import Path
import numpy as np, pandas as pd, torch, ultralytics
from PIL import Image
from ultralytics import YOLO
from sklearn.model_selection import StratifiedGroupKFold
T_START = time.time()
from ultralytics.utils import USER_CONFIG_DIR
if PKG_DIR and not (USER_CONFIG_DIR / 'Arial.ttf').exists():
    shutil.copy(f'{PKG_DIR}/Arial.ttf', USER_CONFIG_DIR / 'Arial.ttf')   # font for plots (no download offline)
N_GPU = torch.cuda.device_count()
print('ultralytics', ultralytics.__version__, '| torch', torch.__version__, '| GPUs:',
      [torch.cuda.get_device_name(i) for i in range(N_GPU)])
if os.path.exists('/kaggle/input') and N_GPU == 0 and os.environ.get('SMOKE_TEST') != '1':
    raise RuntimeError('No GPU in this Kaggle session: enable "GPU T4" (needs a phone-verified Kaggle account)')
""")

md("## Config")
code(rf"""
FOLDS        = [{FOLDS}]        # folds to train in THIS session (0..4)
N_FOLDS      = 5
MODEL        = 'yolo26m.pt'
IMGSZ        = 960
EPOCHS       = 30
PATIENCE     = 10
BATCH_PER_GPU = 8
SEED         = 42
TTA          = True
CONF         = 0.001
MAX_DET      = 300              # raw per model; pycocotools later keeps top-100 per image & class
BUDGET_H     = 11.0             # Kaggle GPU session limit is 12 h -> keep a safety margin
CLASSES = ['Car', 'Motorcycle', 'Bus', 'Truck', 'Tuktuk', 'Van', 'Pickup', 'Songthaew']
W, H = 352, 288
if os.environ.get('SMOKE_TEST') == '1':          # quick local pipeline check
    MODEL, IMGSZ, EPOCHS, FOLDS, BATCH_PER_GPU = 'yolo26n.pt', 320, 1, [0], 8

KAGGLE = os.path.exists('/kaggle/input')
if KAGGLE:
    found = []
    for root, dirs, files in os.walk('/kaggle/input', followlinks=True):
        if root.count(os.sep) <= 6:
            plog(f'{{root}}: {{sorted(dirs)[:10]}} {{[f for f in files if not f.endswith(".jpg")][:10]}}')
        if 'train.csv' in files:
            found.append(root)
    if not found:                 # data attached as a private dataset that still contains the competition zip
        import zipfile
        zips = [z for z in glob.glob('/kaggle/input/**/*.zip', recursive=True)]
        plog(f'extracting {{zips}}')
        for z in zips:
            zipfile.ZipFile(z).extractall('/tmp/data')
        found = [r for r, d, f in os.walk('/tmp/data') if 'train.csv' in f]
    RAW = Path(sorted(found)[0])
    WORK = Path('/kaggle/working')
else:                           # Colab / local: expects data in ./data/raw (kaggle competitions download ...)
    RAW = Path('data/raw'); WORK = Path('.')
OUT = WORK / 'cv_out'; OUT.mkdir(parents=True, exist_ok=True)
TMP = Path('/tmp/dsde'); TMP.mkdir(parents=True, exist_ok=True)
TRAIN_IMG_DIR = next(p for p in [RAW / 'train' / 'train', RAW / 'train'] if p.is_dir() and any(p.glob('*.jpg')))
TEST_IMG_DIR = next(p for p in [RAW / 'test' / 'test', RAW / 'test'] if p.is_dir() and any(p.glob('*.jpg')))
print(RAW, TRAIN_IMG_DIR, TEST_IMG_DIR, OUT); plog(f'paths {{RAW}} {{TRAIN_IMG_DIR}}')
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
""")

md("## Data, cleaning and camera folds (same rules as the main notebook)")
code(r"""
train_df = pd.read_csv(RAW / 'train.csv')
train_files = sorted(p.name for p in TRAIN_IMG_DIR.glob('*.jpg'))
test_files = sorted(p.name for p in TEST_IMG_DIR.glob('*.jpg'))
cam_of = lambda f: f.split('_')[0]

def box_iou(a, b):
    lt = np.maximum(a[:, None, :2], b[None, :, :2]); rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None] - inter + 1e-9)

clean = train_df[train_df.image_id.isin(train_files)].copy()
for c, lim in [('x1', W), ('x2', W), ('y1', H), ('y2', H)]:
    clean[c] = clean[c].clip(0, lim)
clean = clean[~clean.duplicated(['image_id', 'class_id', 'x1', 'y1', 'x2', 'y2'])]
clean = clean[((clean.x2 - clean.x1) >= 2) & ((clean.y2 - clean.y1) >= 2)]
drop = []
for f, g in clean.groupby('image_id'):
    if len(g) > 1:
        iou = np.triu(box_iou(g[['x1', 'y1', 'x2', 'y2']].values, g[['x1', 'y1', 'x2', 'y2']].values), 1)
        drop += [g.index[j] for i, j in zip(*np.where(iou > 0.9)) if g.class_id.iloc[i] == g.class_id.iloc[j]]
clean = clean.drop(index=list(set(drop)))
print('boxes raw', len(train_df), '-> clean', len(clean))

# StratifiedGroupKFold: group = camera, stratify on the rarest class present in each image
freq = clean.class_id.value_counts()
rarest = {f: g.class_id.iloc[g.class_id.map(freq).values.argmin()] for f, g in clean.groupby('image_id')}
y = np.array([rarest.get(f, 8) for f in train_files]); groups = np.array([cam_of(f) for f in train_files])
fold_of_cam = {}
for k, (_, va) in enumerate(StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(train_files, y, groups)):
    for c in set(groups[va]):
        fold_of_cam[c] = k
# The assignment above depends on the scikit-learn version. For exact reproducibility the folds of the submitted run
# (Kaggle, scikit-learn of the Kaggle image) are fixed here; set USE_FIXED_FOLDS = False to recompute them.
USE_FIXED_FOLDS = True
FIXED_FOLDS = {0: ['1426', '1427', '172'], 1: ['1066', '182', '222'], 2: ['229', '232', '244'],
               3: ['1407', '1437', '180'], 4: ['12', '214', '231']}
if USE_FIXED_FOLDS:
    fold_of_cam = {c: k for k, cams in FIXED_FOLDS.items() for c in cams}
json.dump({k: sorted(c for c, v in fold_of_cam.items() if v == k) for k in range(N_FOLDS)}, open(OUT / 'folds.json', 'w'), indent=1)
cc = pd.crosstab(clean.image_id.map(cam_of).map(fold_of_cam), clean.class_id).reindex(columns=range(8), fill_value=0)
cc.columns = CLASSES; cc.index.name = 'val fold'
print({k: sorted(c for c, v in fold_of_cam.items() if v == k) for k in range(N_FOLDS)})
cc
""")
code(r"""
# write YOLO labels once (all train images), then per-fold image lists with repeat-factor sampling
YOLO_DIR = TMP / 'yolo'
if YOLO_DIR.exists():
    shutil.rmtree(YOLO_DIR)
(YOLO_DIR / 'images').mkdir(parents=True); (YOLO_DIR / 'labels').mkdir(parents=True)
by_img = dict(tuple(clean.groupby('image_id')))

def write(src, stem):
    shutil.copy(TRAIN_IMG_DIR / src, YOLO_DIR / 'images' / f'{stem}.jpg')
    g = by_img.get(src)
    lines = [] if g is None else [f'{r.class_id} {(r.x1 + r.x2) / 2 / W:.6f} {(r.y1 + r.y2) / 2 / H:.6f} {(r.x2 - r.x1) / W:.6f} {(r.y2 - r.y1) / H:.6f}'
                                  for r in g.itertuples()]
    (YOLO_DIR / 'labels' / f'{stem}.txt').write_text('\n'.join(lines))
    return str(YOLO_DIR / 'images' / f'{stem}.jpg')

f_c = clean.groupby('class_id').image_id.nunique() / len(train_files)
r_c = np.maximum(1, np.sqrt(0.15 / f_c)).clip(upper=4)
rs = np.random.default_rng(SEED)
base, reps = {}, {}
for f in train_files:
    base[f] = write(f, Path(f).stem)
    r = r_c[by_img[f].class_id.unique()].max() if f in by_img else 1
    n = int(math.floor(r) + (rs.random() < r - math.floor(r)))
    reps[f] = [write(f, f'{Path(f).stem}__rep{k}') for k in range(1, n)]

for k in range(N_FOLDS):
    tr = [p for f in train_files if fold_of_cam[cam_of(f)] != k for p in [base[f]] + reps[f]]
    va = [base[f] for f in train_files if fold_of_cam[cam_of(f)] == k]
    (YOLO_DIR / f'train_f{k}.txt').write_text('\n'.join(tr)); (YOLO_DIR / f'val_f{k}.txt').write_text('\n'.join(va))
    (YOLO_DIR / f'fold{k}.yaml').write_text(f"path: {YOLO_DIR}\ntrain: train_f{k}.txt\nval: val_f{k}.txt\nnc: 8\nnames: {CLASSES}\n")
    print(f'fold {k}: train {len(tr)} (incl. repeats)  val {len(va)}')
""")

md("## Metric (pycocotools, same as Kaggle) and prediction helpers")
code(r"""
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def coco_map50(gt, pred, files):
    fid = {f: i for i, f in enumerate(files)}
    gt = gt[gt.image_id.isin(fid)]; pred = pred[pred.image_id.isin(fid)]
    ds = {'images': [{'id': i, 'file_name': f, 'width': W, 'height': H} for f, i in fid.items()],
          'categories': [{'id': k, 'name': n} for k, n in enumerate(CLASSES)],
          'annotations': [{'id': k + 1, 'image_id': fid[r.image_id], 'category_id': int(r.class_id),
                           'bbox': [float(r.x1), float(r.y1), float(r.x2 - r.x1), float(r.y2 - r.y1)],
                           'area': float((r.x2 - r.x1) * (r.y2 - r.y1)), 'iscrowd': 0} for k, r in enumerate(gt.itertuples())]}
    dets = [{'image_id': fid[r.image_id], 'category_id': int(r.class_id), 'score': float(r.confidence),
             'bbox': [float(r.x1), float(r.y1), float(r.x2 - r.x1), float(r.y2 - r.y1)]} for r in pred.itertuples()]
    with contextlib.redirect_stdout(io.StringIO()):
        cg = COCO(); cg.dataset = ds; cg.createIndex()
        E = COCOeval(cg, cg.loadRes(dets), 'bbox'); E.evaluate(); E.accumulate(); E.summarize()
    prec = E.eval['precision'][0, :, :, 0, 2]
    per = {CLASSES[k]: (round(float(prec[:, k][prec[:, k] > -1].mean()), 4) if (prec[:, k] > -1).any() else None) for k in range(8)}
    return round(float(E.stats[1]), 4), per

PRED_BATCH = int(os.environ.get('PRED_BATCH', 32))   # lower it on small GPUs

def predict(model, img_dir, files, batch=PRED_BATCH):
    rows = []
    for i in range(0, len(files), batch):
        chunk = [str(img_dir / f) for f in files[i:i + batch]]
        for p, r in zip(chunk, model.predict(chunk, imgsz=IMGSZ, conf=CONF, max_det=MAX_DET, augment=TTA, verbose=False, device=0)):
            b = r.boxes
            for (x1, y1, x2, y2), c, s in zip(b.xyxy.cpu().numpy(), b.cls.cpu().numpy(), b.conf.cpu().numpy()):
                rows.append((Path(p).name, int(c), float(s), float(x1), float(y1), float(x2), float(y2)))
    return pd.DataFrame(rows, columns=['image_id', 'class_id', 'confidence', 'x1', 'y1', 'x2', 'y2'])
""")

md("## Train the folds of this session")
code(r"""
fold_times = []
for k in FOLDS:
    fdir = OUT / f'fold{k}'
    if (fdir / 'test_pred.csv').exists():
        print(f'fold {k}: already done'); continue
    elapsed = (time.time() - T_START) / 3600
    if fold_times and elapsed + 1.15 * max(fold_times) > BUDGET_H:
        print(f'fold {k}: SKIPPED (elapsed {elapsed:.2f} h, last fold took {max(fold_times):.2f} h) -> run it in another session')
        continue
    t0 = time.time()
    run = TMP / 'runs' / f'fold{k}'
    if (fdir / 'best.pt').exists():            # weights already there (e.g. submitted weights) -> only predict
        print(f'fold {k}: using existing {fdir / "best.pt"}')
        train_this = False
    else:
        train_this = True
    args = dict(data=str(YOLO_DIR / f'fold{k}.yaml'), imgsz=IMGSZ, epochs=EPOCHS, patience=PATIENCE, seed=SEED,
                deterministic=True, cos_lr=True, close_mosaic=10, cache='ram', workers=4, plots=True,
                project=str(run.parent), name=run.name, exist_ok=True)
    if train_this:
        try:
            YOLO(MODEL).train(batch=BATCH_PER_GPU * max(N_GPU, 1), device=list(range(N_GPU)) if N_GPU > 1 else 0, **args)
        except Exception as e:                       # multi-GPU (DDP) failure -> fall back to one GPU
            print('multi-GPU training failed, retrying on 1 GPU:', repr(e)[:300])
            YOLO(MODEL).train(batch=BATCH_PER_GPU, device=0, **args)
        fdir.mkdir(parents=True, exist_ok=True)
        for fn in ['weights/best.pt', 'results.csv', 'args.yaml', 'confusion_matrix_normalized.png', 'results.png']:
            if (run / fn).exists():
                shutil.copy(run / fn, fdir / Path(fn).name)
    model = YOLO(str(fdir / 'best.pt'))
    val_files = [f for f in train_files if fold_of_cam[cam_of(f)] == k]
    vp = predict(model, TRAIN_IMG_DIR, val_files); vp.to_csv(fdir / 'val_pred.csv', index=False)
    m, per = coco_map50(train_df, vp, val_files)                  # raw labels, like Kaggle
    predict(model, TEST_IMG_DIR, test_files).to_csv(fdir / 'test_pred.csv', index=False)
    hours = (time.time() - t0) / 3600; fold_times.append(hours)
    meta = {'fold': k, 'val_cams': sorted(c for c, v in fold_of_cam.items() if v == k), 'val_map50': m, 'val_ap': per,
            'hours': round(hours, 2), 'model': MODEL, 'imgsz': IMGSZ, 'epochs': EPOCHS, 'tta': TTA, 'gpus': N_GPU}
    json.dump(meta, open(fdir / 'metrics.json', 'w'), indent=2)
    print(json.dumps(meta)); plog(json.dumps(meta))
""")

md("## Out-of-fold CV score and WBF ensemble (uses every fold found in `cv_out/` and in attached inputs)")
code(r"""
from ensemble_boxes import weighted_boxes_fusion
fold_dirs = {}
for d in sorted(glob.glob(str(OUT / 'fold*'))) + sorted(glob.glob('/kaggle/input/**/cv_out/fold*', recursive=True)):
    d = Path(d)
    if (d / 'test_pred.csv').exists():
        fold_dirs.setdefault(int(d.name[4:]), d)
print('folds available:', sorted(fold_dirs))
summary = [json.load(open(d / 'metrics.json')) for _, d in sorted(fold_dirs.items())]
display(pd.DataFrame(summary)[['fold', 'val_cams', 'val_map50', 'hours']])
if fold_dirs:
    oof = pd.concat([pd.read_csv(d / 'val_pred.csv') for d in fold_dirs.values()])
    oof_files = [f for f in train_files if fold_of_cam[cam_of(f)] in fold_dirs]
    cv, cv_ap = coco_map50(train_df, oof, oof_files)
    print(f'OUT-OF-FOLD CV mAP@50 over {len(oof_files)} images / folds {sorted(fold_dirs)}: {cv}')
    print(cv_ap)
""")
code(r"""
def wbf(preds, iou_thr=0.55, skip=CONF):
    by = [dict(tuple(p.groupby('image_id'))) for p in preds]
    rows = []
    for f in test_files:
        bl, sl, ll = [], [], []
        for d in by:
            g = d.get(f)
            if g is None:
                bl.append(np.zeros((0, 4))); sl.append(np.zeros(0)); ll.append(np.zeros(0)); continue
            bl.append(np.clip(g[['x1', 'y1', 'x2', 'y2']].values / [W, H, W, H], 0, 1)); sl.append(g.confidence.values); ll.append(g.class_id.values)
        if not any(len(s) for s in sl):
            continue
        b, s, l = weighted_boxes_fusion(bl, sl, ll, iou_thr=iou_thr, skip_box_thr=skip, conf_type='avg')
        for (x1, y1, x2, y2), sc, lb in zip(b * [W, H, W, H], s, l):
            rows.append((f, int(lb), float(sc), x1, y1, x2, y2))
    return pd.DataFrame(rows, columns=['image_id', 'class_id', 'confidence', 'x1', 'y1', 'x2', 'y2'])

if fold_dirs:
    ens = wbf([pd.read_csv(d / 'test_pred.csv') for _, d in sorted(fold_dirs.items())])
    ens = ens.round({'confidence': 5, 'x1': 2, 'y1': 2, 'x2': 2, 'y2': 2})
    ens = ens[(ens.x2 > ens.x1) & (ens.y2 > ens.y1)]
    ens = ens.sort_values(['image_id', 'confidence'], ascending=[True, False]).groupby('image_id').head(300).reset_index(drop=True)
    ens.insert(0, 'id', range(len(ens)))
    name = f"{Path(MODEL).stem}_{IMGSZ}_cv{len(fold_dirs)}fold_wbf.csv"
    ens.to_csv(OUT / name, index=False)
    print('saved', OUT / name, len(ens), 'rows,', ens.image_id.nunique(), 'images')
print(f'total time {(time.time() - T_START) / 3600:.2f} h')
""")

nb = nbf.v4.new_notebook(); nb['cells'] = cells
nb['metadata'] = {'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}, 'language_info': {'name': 'python'}}
nbf.write(nb, sys.argv[1]); print('wrote', sys.argv[1], 'folds', FOLDS)
