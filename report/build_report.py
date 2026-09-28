"""Build the midterm report (report/midterm_report.docx + .pdf).

Numbers come from the data (data/raw) and from report/results.json (experiment summary).
Run from the midterm/ folder:  python report/build_report.py
"""
import json
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / 'report' / 'figures'
RES = json.loads((ROOT / 'report' / 'results.json').read_text(encoding='utf-8'))
CLASSES = ['Car', 'Motorcycle', 'Bus', 'Truck', 'Tuktuk', 'Van', 'Pickup', 'Songthaew']

# ---------------------------------------------------------------- data stats
raw = ROOT / 'data' / 'raw'
tr = pd.read_csv(raw / 'train.csv')
tr['cam'] = tr.image_id.str.split('_').str[0]
n_train_img = len(list((raw / 'train' / 'train').glob('*.jpg')))
n_test_img = len(list((raw / 'test' / 'test').glob('*.jpg')))
test_cams = sorted({p.name.split('_')[0] for p in (raw / 'test' / 'test').glob('*.jpg')})
train_cams = sorted({p.name.split('_')[0] for p in (raw / 'train' / 'train').glob('*.jpg')})
cls_boxes = tr.class_id.value_counts().sort_index()
cls_imgs = tr.groupby('class_id').image_id.nunique()
bw, bh = tr.x2 - tr.x1, tr.y2 - tr.y1

# ---------------------------------------------------------------- helpers
doc = Document()
sec = doc.sections[0]
sec.page_height, sec.page_width = Cm(29.7), Cm(21.0)
for side in ('left_margin', 'right_margin'):
    setattr(sec, side, Cm(2.2))
sec.top_margin = sec.bottom_margin = Cm(2.0)
st = doc.styles['Normal']
st.font.name = 'Calibri'
st.font.size = Pt(11)
st.paragraph_format.space_after = Pt(6)
st.paragraph_format.line_spacing = 1.15

fig_no = [0]
tab_no = [0]


def H(text, level=1):
    return doc.add_heading(text, level=level)


def P(text='', bold=False, italic=False, align=None, size=None):
    p = doc.add_paragraph()
    # **bold** segments inside text
    parts = text.split('**')
    for i, part in enumerate(parts):
        r = p.add_run(part)
        r.bold = bold or (i % 2 == 1)
        r.italic = italic
        if size:
            r.font.size = Pt(size)
    if align:
        p.alignment = align
    return p


def B(items, style='List Bullet'):
    for it in items:
        p = doc.add_paragraph(style=style)
        parts = it.split('**')
        for i, part in enumerate(parts):
            p.add_run(part).bold = i % 2 == 1


def caption(prefix, counter, text):
    counter[0] += 1
    p = doc.add_paragraph()
    r = p.add_run(f'{prefix} {counter[0]}: {text}')
    r.italic = True
    r.font.size = Pt(9.5)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    return counter[0]


def FIGURE(fn, text, width=16):
    path = FIG / fn
    if not path.exists():
        P(f'[missing figure: {fn}]', italic=True)
        return
    doc.add_picture(str(path), width=Cm(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption('Figure', fig_no, text)


def shade(cell, hex_color):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), hex_color)
    tcPr.append(shd)


def TABLE(header, rows, text, widths=None, bold_rows=()):
    caption('Table', tab_no, text)
    t = doc.add_table(rows=1, cols=len(header))
    t.style = 'Table Grid'
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(header):
        c = t.rows[0].cells[i]
        c.text = ''
        r = c.paragraphs[0].add_run(str(h))
        r.bold = True
        r.font.size = Pt(9.5)
        shade(c, 'D9E2F3')
    for ri, row in enumerate(rows):
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ''
            r = cells[i].paragraphs[0].add_run(str(v))
            r.font.size = Pt(9.5)
            r.bold = ri in bold_rows
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Cm(w)
    doc.add_paragraph()
    return t


def page_break():
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


best = RES['experiments'][RES['best']]
final = RES.get('final', {})
fmt = lambda x: '-' if x is None else f'{x:.4f}'

# ---------------------------------------------------------------- title page
for _ in range(6):
    doc.add_paragraph()
P('2110531 Data Science and Data Engineering Tools (2026/1)', align=WD_ALIGN_PARAGRAPH.CENTER, size=13)
P('Take-Home Midterm Exam Report', bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, size=22)
P('Bangkok CCTV Vehicle Detection', bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, size=18)
doc.add_paragraph()
P(f"Name: {RES['student']['name']}", align=WD_ALIGN_PARAGRAPH.CENTER, size=12)
P(f"Student ID: {RES['student']['id']}", align=WD_ALIGN_PARAGRAPH.CENTER, size=12)
P(f"Kaggle username: {RES['student']['kaggle']}", align=WD_ALIGN_PARAGRAPH.CENTER, size=12)
doc.add_paragraph()
P(f"Source code / data / weights: {RES['links']['code']}", align=WD_ALIGN_PARAGRAPH.CENTER, size=10)
P(f"Model weights (Google Drive): {RES['links']['weights']}", align=WD_ALIGN_PARAGRAPH.CENTER, size=10)
page_break()

# ---------------------------------------------------------------- 1 introduction
H('Chapter 1: Introduction')
P('The task is multi-class object detection of vehicles in Bangkok traffic-camera (CCTV) images. '
  'The images were scraped automatically every 30 minutes from the Bangkok Metropolitan Administration (BMA) '
  'traffic platform (cpudapp.bangkok.go.th/bmatraffic) between 25/08/2026 06:00 and 31/08/2026 21:00. '
  'After a quality screening, 20 cameras were kept. Every vehicle has to be localised with a bounding box and '
  'assigned to one of eight classes: Car, Motorcycle, Bus, Truck, Tuktuk, Van, Pickup and Songthaew.')
P('Submissions are scored on Kaggle with **mAP@50**: the COCO-style average precision at IoU 0.50, '
  'computed with pycocotools for each class and averaged over the eight classes. Because every class has the '
  'same weight, rare classes such as Songthaew or Tuktuk affect the score as much as Car does.')
P('Objectives of this work:')
B(['explore and clean the data, and build a validation scheme that reflects the hidden test set;',
   'fine-tune a modern pretrained detector (YOLO26 from the Ultralytics framework) and evaluate it with the '
   'same metric as Kaggle;',
   'analyse the errors and describe how the model could be improved.'])
P('All code is in a single Jupyter notebook (vehicle_detection.ipynb) that runs end-to-end: EDA, cleaning, '
  'split, training, evaluation, visualisation and submission. No generative-AI / VLM API was used for '
  'prediction, so there is no API cost; training ran on a local laptop GPU (NVIDIA RTX 3050 Ti, 4 GB).')

# ---------------------------------------------------------------- 2 data
H('Chapter 2: Data Preparation')
P('The analysis follows the data-preparation steps taught in the course (Week 2 slides and Lab1-Lab4): '
  '(1) examine the data set, (2) understand / narrow down the features and check for leakage, '
  '(3) handle outliers and skewness, and (4) build the train / validation split with a fixed random seed.')

H('2.1 Examining the data set', 2)
TABLE(['Item', 'Value'], [
    ['Train images', f'{n_train_img:,} ({len(train_cams)} cameras)'],
    ['Test images', f'{n_test_img:,} ({len(test_cams)} cameras); 997 are listed in sample_submission.csv'],
    ['Image size', '352 x 288 pixels (all images)'],
    ['Annotated boxes (train)', f'{len(tr):,} in {tr.image_id.nunique():,} images; {n_train_img - tr.image_id.nunique()} images have no vehicle'],
    ['Boxes per image', f'mean {len(tr) / n_train_img:.1f}, max {tr.groupby("image_id").size().max()}'],
    ['Box size (w x h)', f'median {bw.median():.0f} x {bh.median():.0f} px, 25% of boxes are smaller than {bw.quantile(.25):.0f} x {bh.quantile(.25):.0f} px'],
    ['Submission format', 'id, image_id, class_id, confidence, x1, y1, x2, y2 (one row per box)'],
], 'Dataset summary', widths=[4.5, 12])
TABLE(['Column', 'Type', 'Description', 'Missing'], [
    ['id', 'int', 'row id - identifier only, not a feature', 0],
    ['image_id', 'str', '<camera>_<YYYYMMDD>_<HHMMSS>.jpg; camera, date, hour and day/night are derived from it', 0],
    ['class_id', 'int (nominal)', 'target class 0-7', 0],
    ['x1, y1', 'int', 'top-left corner of the box [px]', 0],
    ['x2, y2', 'int', 'bottom-right corner of the box [px]', 0],
], 'Data dictionary of train.csv (27,396 rows x 7 columns)', widths=[2.2, 2.6, 10, 1.6])
P('**Out-of-range values and miscodes.** No missing values; all class ids are in 0-7; no inverted boxes and no box '
  'outside the image; every image_id exists on disk and all images are 352 x 288. Found: 25 exact duplicate boxes, '
  '15 boxes with a side < 2 px, and 102 training images without any vehicle (kept as negative examples).')
H('Target distribution', 3)
TABLE(['ID', 'Class', 'Boxes', 'Share', 'Images containing class', 'Imbalance vs Car'],
      [[k, CLASSES[k], f'{cls_boxes[k]:,}', f'{cls_boxes[k] / len(tr):.2%}', f'{cls_imgs[k]:,}',
        f'{cls_boxes[0] / cls_boxes[k]:.1f}x'] for k in range(8)],
      'Class distribution in the training set', widths=[1.1, 2.8, 2, 2, 4, 3.2])
FIGURE('eda_class_dist.png', 'Boxes per class (log scale) and fraction of training images that contain each class')
P('The target is heavily imbalanced: Car is 64.8% of all boxes while Songthaew is 0.43% (152 times fewer). Because '
  'mAP@50 averages the AP of the eight classes equally, the rare classes matter as much as Car.')

H('2.2 Feature understanding, association with the target and leakage', 2)
P(f'**Camera.** Train uses cameras {", ".join(train_cams)}; test uses {", ".join(test_cams)}. '
  '**The two sets are disjoint**, so the model must generalise to unseen viewpoints.')
FIGURE('eda_camera.png', 'Images per camera (train vs test) and boxes per camera x class')
st = RES['eda_stats']
P(f"**Chi-square test of camera x class** (slide appendix): chi2 = {st['chi2_cam']:,}, dof = 98, p < 1e-300, "
  f"Cramer's V = {st['v_cam']:.3f}. H0 (same class distribution for every camera) is rejected: the vehicle mix depends "
  "strongly on the camera. The standardised residuals show, for example, that 91 of the 117 Songthaews come from camera "
  "1426 (residual +53), Pickups are over-represented at cameras 232 and 222 and Tuktuks at 244, 229 and 172.")
FIGURE('eda_chi2_resid.png', 'Standardised chi-square residuals, camera x class (red = more boxes than expected)', width=14)
P(f"**Time of day.** Frames cover 06:00-21:00 every 30 minutes. Street lights keep night frames bright, so night was "
  f"defined by clock time (after 18:30, about 17% of frames). The class mix also changes at night (chi-square "
  f"p = {st['p_night']}, Cramer's V = {st['v_night']:.3f}); e.g. Songthaew rises from 0.27% of boxes by day to "
  "1.53% at night and Tuktuk from 0.84% to 1.86%.")
FIGURE('eda_brightness.png', 'Mean brightness vs hour of day', width=12)
FIGURE('eda_time.png', 'Traffic density by hour and class share by day / night')
P(f"**Leakage check.** In Lab1, columns that leak information were removed. Here the leak is at the image level: "
  f"consecutive frames of the same camera differ by a median of {st['mad_same']} grey levels, versus {st['mad_cross']} "
  f"for frames of different cameras ({st['mad_ratio']}x more similar). A random train / validation split would put "
  "almost identical scenes on both sides and measure memorisation instead of generalisation. Therefore **whole cameras "
  "are held out** for validation (Section 2.6).")
FIGURE('eda_leakage.png', 'Frame similarity: same camera (consecutive frames) vs different cameras', width=13)
P('**Train vs test distribution.** The test cameras have a similar share of night frames (17.6% vs 17.1%) and a '
  'slightly darker mean brightness (123 vs 132).')
FIGURE('eda_train_vs_test.png', 'Brightness and hour-of-day distributions of train and test images')

H('2.3 Outliers and skewness', 2)
P(f"Box sizes are tiny and strongly right-skewed: the box area has skew {st['skew_area']} and kurtosis "
  f"{st['kurt_area']}; after a log transform the skew is {st['skew_log']} (Lab4 approach). This motivates upscaling "
  "the images (imgsz 640 -> 960) and a multi-scale detector.")
FIGURE('eda_box_size.png', 'Boxes per image, box size per class and box width vs height')
FIGURE('eda_area_skew.png', 'Box area before and after log transform')
TABLE(['Class', 'Boxes', 'Median area', 'log-area IQR outliers', 'Median h/w', 'h/w IQR outliers'],
      RES['outlier_table'], 'Outliers per class (1.5 x IQR rule)', widths=[2.8, 1.8, 2.5, 3.4, 2.4, 3])
P('The extremes were inspected visually before deciding what to do. The largest boxes are real vehicles very close '
  'to camera 1426 (6 boxes are almost full height) and are kept. The extreme aspect ratios are annotation errors: '
  'slivers such as 1 x 8 or 2 x 16 px with no vehicle inside (195 boxes have a side <= 2 px). Boxes with a side '
  '< 2 px are removed; truncating coordinates, as done for numeric features in Lab4, is not meaningful for boxes.')
FIGURE('eda_extreme_boxes.png', 'Most extreme boxes: largest area (top), tallest (middle), widest (bottom)')

H('2.4 Visual inspection', 2)
FIGURE('eda_train_samples.png', 'Preview: one daytime frame per training camera with ground-truth boxes')
FIGURE('eda_train_night_samples.png', 'Preview: night frames of training cameras. Training uses all frames (day and night); only this preview is split by time')
FIGURE('eda_test_samples.png', 'Test cameras (day and night); these viewpoints never appear in training')
FIGURE('eda_crops.png', 'Examples of rare classes (crops). Van / Pickup / Truck are hard to separate even for a human', width=13)
P('**sample_submission.csv uses different ids from the test file names.** It lists 997 long ids '
  '(dataset_<cam>_annotated_coco1.0_<cam>_<Thai location name>_<date>_<time>.jpg), while the 1,013 test images are '
  'named <cam>_<date>_<time>.jpg. The Kaggle solution uses the file names (Section 4.3), so predictions are written '
  'for all 1,013 test images with the file name as image_id.')

H('2.5 Data cleaning', 2)
P('Cleaning is applied to the **training labels only**. Validation is always scored against the raw labels, '
  'the same way Kaggle scores the raw test labels.')
cl = RES['cleaning']
TABLE(['Step', 'Count'], [[k, v] for k, v in cl.items()], 'Data-cleaning log', widths=[11, 3])
P('280 pairs of boxes overlap with IoU > 0.9 but have different classes (mostly Car vs Truck and '
  'Bus vs Truck): the same vehicle was labelled twice. They were kept because the test labels probably '
  'contain the same pattern, and a model that outputs both classes at lower confidence can still match both.')

H('2.6 Train / validation split and dataset construction', 2)
P(f"To mimic the unseen-camera test set and avoid the leakage shown in 2.2, **{len(RES['val_cams'])} whole cameras "
  f"({', '.join(RES['val_cams'])}) were held out for validation**. As a stratification over cameras, the notebook "
  "tries all 1,365 combinations of 4 cameras out of 15 and keeps the one where every class has at least 10 validation "
  "boxes and the validation share of each class is closest to 4/15. The random seed is fixed (42) everywhere.")
TABLE(['Class', 'Train boxes', 'Val boxes', 'Val share', 'Train %', 'Val %'],
      [[c, *RES['split_table'][c]] for c in CLASSES],
      'Camera-held-out split after cleaning: 2,194 train / 797 val images (class proportions compared as in Lab3)',
      widths=[3, 2.4, 2.4, 2.2, 2.2, 2.2])
P('**Repeat-factor sampling (RFS).** mAP weights all classes equally, so images that contain rare classes '
  'are repeated during training (LVIS-style): for class c with image frequency f_c, r_c = max(1, sqrt(t / f_c)) '
  'with t = 0.15, and each image is repeated r = max_c r_c times with stochastic rounding. Songthaew images are '
  'repeated about 2.8x, Tuktuk 1.4x and Van 1.2x. Repeats are only added to the training list, never to validation.')
P('Labels were converted to the YOLO format (class x_center y_center width height, normalised) and '
  'the image lists were written as train.txt / val.txt / full.txt, which Ultralytics reads directly.')

H('2.7 Summary: findings and decisions', 2)
TABLE(['Finding', 'Decision'], [
    ['Test cameras never appear in training; same-camera frames are near-duplicates', 'Hold out whole cameras for validation'],
    ['Class mix depends on camera and time (chi-square p ~ 0)', 'Every class must be present in val; RFS for rare classes'],
    ['Car : Songthaew = 152 : 1', 'Repeat-factor sampling; per-class AP analysis'],
    ['Tiny, right-skewed box sizes', 'Upscale to imgsz 960; multi-scale detector'],
    ['Sliver boxes are label noise; duplicates', 'Remove side < 2 px and exact / near duplicates'],
    ['17% night frames in both train and test', 'Report day / night mAP; HSV augmentation'],
], 'EDA findings and the resulting decisions', widths=[9, 7.5])

# ---------------------------------------------------------------- 3 model
H('Chapter 3: Model')
H('3.1 Choice of model', 2)
P('The course slides (Week 5, Object Detection) describe the change from two-stage CNNs (Faster R-CNN) to '
  'one-stage YOLO detectors, transformer detectors (DETR / RT-DETR) and, most recently, end-to-end NMS-free '
  'YOLO26. The course example notebook fine-tunes YOLOv8 with Ultralytics. This work uses the newest model in '
  'that family, **YOLO26**, fine-tuned from COCO-pretrained weights. YOLO11 and RT-DETR can be selected with a '
  'single config line for comparison.')
TABLE(['Property', 'YOLO26s (used)'], [
    ['Framework', 'Ultralytics 8.4.163, PyTorch 2.8.0 + CUDA 12.6'],
    ['Parameters / FLOPs', '9.95 M parameters, 22.8 GFLOPs at 640 (8-class head)'],
    ['Pretraining', 'COCO 2017 detection (80 classes, ~118k images)'],
    ['Backbone', 'CSP-style CNN with C3k2 blocks, SPPF (spatial pyramid pooling) and C2PSA (attention) block'],
    ['Neck', 'PAN/FPN feature pyramid, fuses strides 8 / 16 / 32 (important for small vehicles)'],
    ['Head', 'Anchor-free, end-to-end (one-to-one) head: no NMS at inference; no DFL'],
    ['Transfer', "4 of 8 class rows of the head (car, motorcycle, bus, truck) were initialised from the matching COCO classes"],
], 'YOLO26s summary', widths=[4, 12.5])
P('YOLO is a one-stage detector: a single CNN pass predicts, for every location of three feature maps, a box '
  'and class scores. YOLO26 trains with a one-to-one assignment so each object gets a single prediction, which '
  'removes the non-maximum-suppression step, and it drops the Distribution Focal Loss. According to the '
  'Ultralytics documentation it also uses progressive loss balancing (ProgLoss) and small-target-aware label '
  'assignment (STAL), which suits this dataset because most vehicles are tiny.')

H('3.2 Training setup', 2)
TABLE(['Hyper-parameter', 'Value'], [
    ['Input size', f"{best['imgsz']} (images are upscaled from 352 x 288 with letterbox)"],
    ['Epochs / early stopping', f"{best['epochs']} max, patience 30; the best-val-mAP checkpoint is kept"],
    ['Batch size', f"{best['batch']} (limited by 4 GB VRAM)"],
    ['Optimizer', 'Ultralytics "auto" -> AdamW (lr 0.000833, momentum 0.9, weight decay 0.0005), cosine LR schedule, warm-up 3 epochs'],
    ['Augmentation', 'Mosaic (off for the last 10 epochs), random scale 0.5, translate 0.1, horizontal flip 0.5, HSV jitter (h 0.015, s 0.7, v 0.4), random erasing 0.4'],
    ['Class balance', 'Repeat-factor sampling (Section 2.6)'],
    ['Seed', '42, deterministic=True'],
    ['Hardware', 'NVIDIA GeForce RTX 3050 Ti Laptop GPU (4 GB), 32 GB RAM, Windows 11'],
], 'Training configuration', widths=[4, 12.5])
H('3.3 Inference and post-processing', 2)
B(['Confidence threshold 0.001: low-confidence boxes only extend the precision-recall curve and never '
   'reduce AP, so a very low threshold gives the highest mAP.',
   f"max_det = {best.get('max_det', 100)} boxes per image (pycocotools keeps at most 100 per image and class).",
   'Boxes are written in absolute pixel coordinates (x1, y1, x2, y2); degenerate boxes are removed; image_id is the '
   'test file name; images without detections get no row and no dummy boxes are added (Data-page rule).'])
H('3.4 Evaluation', 2)
P('The notebook re-implements the Kaggle metric with pycocotools (COCOeval, IoU = 0.50, area = all, maxDets = 100) '
  'on the held-out cameras, using the raw labels. This is the number used to choose models. Ultralytics\' own '
  'mAP50 is a bit higher because it uses the cleaned labels.')

# ---------------------------------------------------------------- 4 results
H('Chapter 4: Results')
H('4.1 Experiments (validation = held-out cameras)', 2)
rows = []
for name, e in RES['experiments'].items():
    rows.append([name, e['desc'], fmt(e.get('val_map50')), fmt(e.get('kaggle_public'))])
TABLE(['Run', 'Description', 'Val mAP@50', 'Kaggle public'], rows, 'Experiment summary',
      widths=[3.6, 8, 2.4, 2.4], bold_rows=[list(RES['experiments']).index(RES['best'])])
H('4.2 Per-class AP of the best model', 2)
TABLE(['Class'] + CLASSES, [['AP@50'] + [f"{best['val_ap'][c]:.3f}" for c in CLASSES]],
      f"Per-class AP@50 on validation cameras ({RES['best']})")
FIGURE(RES['figs']['per_class'], 'Per-class AP@50 (validation)', width=13)
FIGURE(RES['figs']['curves'], 'Training curves (losses and validation mAP)')
FIGURE(RES['figs']['confusion'], 'Normalised confusion matrix on the validation cameras', width=13)
FIGURE(RES['figs']['gt_pred'], 'Ground truth (left) vs prediction with confidence > 0.3 (right) on validation cameras')
FIGURE(RES['figs']['test'], 'Predictions on the unseen test cameras')
H('4.3 Kaggle submission', 2)
TABLE(['Submitted file', 'image_id format', 'Public mAP@50'], RES['kaggle_history'], 'Kaggle submission history', widths=[5.2, 7.5, 3])
P('The first two submissions scored 0.000 although the model was the same: they used the long ids copied from '
  'sample_submission.csv, which do not exist in the solution file. Re-submitting the same predictions with the test '
  'file names gave 0.516, close to the validation score of the same model (0.525), which confirms that the '
  'camera-held-out validation is a reliable estimate of the leaderboard. Training on all 15 cameras raised the '
  'public score to **0.548**.')
P(f"Final submission file: {final.get('file', '-')}. Public leaderboard mAP@50: {fmt(final.get('kaggle_public'))}.")
P('Model weights: weights/yolo26s_960_full.pt (final, all cameras) and weights/yolo26s_960.pt (trained without the 4 validation cameras).')
P('[Insert screenshot of the Kaggle submission page / leaderboard here]', italic=True)

# ---------------------------------------------------------------- 5 discussion
H('Chapter 5: Discussion')
for title, items in RES['discussion'].items():
    H(title, 2)
    B(items)

# ---------------------------------------------------------------- 6 conclusion
H('Chapter 6: Conclusion')
for para in RES['conclusion']:
    P(para)

H('Appendix: How to reproduce', 1)
B(['conda create -n dsde python=3.12; pip install torch==2.8.0 torchvision==0.23.0 --index-url '
   'https://download.pytorch.org/whl/cu126; pip install -r requirements.txt',
   'kaggle competitions download -c 2110531-dsde-2026-1 -p data/raw, then unzip into data/raw',
   'Open vehicle_detection.ipynb from the midterm/ folder and run all cells. The settings are in the Config cell '
   '(or set environment variables, e.g. MODEL, IMGSZ, EPOCHS, FULL_TRAIN=1).',
   'To reproduce the submission without training: SKIP_TRAIN=1 WEIGHTS=<path to submitted .pt> (and the same '
   'IMGSZ / TTA as in the submission .json file).'])

out = ROOT / 'report' / 'midterm_report.docx'
doc.save(out)
print('saved', out)
try:
    from docx2pdf import convert
    convert(str(out), str(out.with_suffix('.pdf')))
    print('saved', out.with_suffix('.pdf'))
except Exception as e:  # Word not available
    print('PDF conversion skipped:', e)
