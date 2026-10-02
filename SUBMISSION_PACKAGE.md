# Submission package — 2110531 DSDE Take-Home Midterm (2026/1)

Chinnakrit Ongsritrakul (6870061721) · Kaggle team `6870061721_Chinnakrit`

The exam asks for **Part 1: (1) source code, (2) uploaded csv (result), (3) prepared data, (4) model weights** and
**Part 2: report (Word & PDF)**. The result must be reproducible by the submitted code and close to the Kaggle score.

Large files (weights, prepared data) are on Google Drive: `<Google Drive link>`. Everything else is in this GitHub repo: `<GitHub link>`.

## Part 1

### (1) Source code
| File | Purpose |
|---|---|
| `vehicle_detection.ipynb` | EDA, cleaning, camera-held-out split, YOLO26s experiments, evaluation (pycocotools), visualisation, submission |
| `cv_kfold_kaggle.ipynb` + `kaggle_kernel/` | **v1**: YOLO26m camera 5-fold CV + TTA + WBF (Kaggle GPU) |
| `cv_kfold_kaggle_v2.ipynb` + `kaggle_kernel_v2/` | **v2**: same as v1 + cross-camera copy-paste of rare classes + photometric variants |
| `tools/build_cv_nb_v1.py`, `tools/build_cv_nb_v2.py` | scripts that generate the two CV notebooks |
| `experiments/ensemble_wbf.py` | fuses fold test predictions with WBF → submission CSV (used for every final submission) |
| `experiments/postproc_eval.py`, `cv_analysis.py`, `label_audit.py` | post-processing study, CV figures, label audit |
| `report/build_report.py` | builds the report from `report/results.json` |
| `requirements.txt`, `README.md` | environment and how to run |

Git tag `v1-yolo26m-cv5-0.60457` marks the code state of the first YOLO26m submission.

### (2) Uploaded CSV files (`submissions/`)
| File | Model | Kaggle public |
|---|---|---|
| `yolo26s_960_tta.csv` | YOLO26s, 11 cameras | 0.51648 |
| `yolo26s_960_full_tta.csv` | YOLO26s, 15 cameras | 0.54814 |
| `yolo26m_960_cv5fold_wbf.csv` | v1 5 folds, WBF iou 0.55 / avg | 0.60457 |
| `yolo26m_960_cv5fold_wbf_iou07.csv` | v1 5 folds, WBF iou 0.7 / box_and_model_avg | 0.61367 |
| *(to be added)* v2 and v1+v2 ensembles | | |

Each CSV has a `.json` with its settings where produced by a notebook.

### (3) Prepared data (Google Drive)
| File | Content |
|---|---|
| `prepared_data_v1.zip` | YOLO-format dataset of v1: cleaned labels, repeat-factor copies, fold lists `train_f{k}.txt` / `val_f{k}.txt`, `fold{k}.yaml` (relative paths) |
| `prepared_data_v2.zip` | same for v2: + 500 copy-paste images per fold (`*__cp{k}_*`), photometric repeat copies, 8 sliver boxes removed |

Both are regenerated automatically by the notebooks from the raw Kaggle data (`data/raw`, seed 42).

### (4) Model weights (Google Drive)
| File | Content |
|---|---|
| `weights_v1_and_yolo26s.zip` | `yolo26m_960_cv_fold{0..4}.pt` (v1) + `yolo26s_960_full.pt`, `yolo26s_960.pt`, `yolo26s_640.pt` |
| `weights_v2.zip` *(after the v2 run)* | `yolo26m_960_cv_v2_fold{0..4}.pt` |

## Part 2 — report
`report/midterm_report.docx` and `report/midterm_report.pdf` (Chapters 1–6 + appendix).

## How the TA can reproduce the final CSV
1. `conda create -n dsde python=3.12`, `pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126`, `pip install -r requirements.txt`
2. `kaggle competitions download -c 2110531-dsde-2026-1 -p data/raw` and unzip.
3. **Without training (minutes):** copy the weights to `cv_out/fold{k}/best.pt` (v1) or `cv_out_v2/fold{k}/best.pt` (v2) and run the
   notebook → `fold{k}/test_pred.csv`; then `python experiments/ensemble_wbf.py OUT.csv 0.7 box_and_model_avg <pred dirs>`.
   Verified locally: every fold score within 0.0002 of the Kaggle run.
4. **With training (~8–10 h GPU):** run the notebook on Kaggle (`kaggle kernels push -p kaggle_kernel[_v2] --accelerator NvidiaTeslaT4`)
   with the datasets `chinnakrit/dsde-midterm-offline-pkgs` and `chinnakrit/dsde-midterm-data`, or on any GPU machine.
