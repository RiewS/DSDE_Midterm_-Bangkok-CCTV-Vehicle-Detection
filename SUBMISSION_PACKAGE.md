# Submission package — 2110531 DSDE Take-Home Midterm (2026/1)

Chinnakrit Ongsritrakul (6870061721) · Kaggle team `6870061721_Chinnakrit`

**Final Kaggle submission:** `submissions/yolo26m_960_v1v2_10models_wbf07.csv` — public mAP@50 **0.62963**
(10 YOLO26m fold models = v1 + v2 camera 5-fold CV, TTA, Weighted Boxes Fusion IoU 0.7 / box_and_model_avg).

The exam asks for **Part 1: (1) source code, (2) uploaded csv (result), (3) prepared data, (4) model weights** and
**Part 2: report (Word & PDF)**. The result must be reproducible by the submitted code and close to the Kaggle score.

* GitHub (code, CSVs, report, experiment outputs): https://github.com/RiewS/DSDE_Midterm_-Bangkok-CCTV-Vehicle-Detection
* Google Drive (weights, prepared data, source-code zip): `<Google Drive link>`

## Part 1

### (1) Source code
| File | Purpose |
|---|---|
| `vehicle_detection.ipynb` | EDA, cleaning, camera-held-out split, YOLO26s experiments, evaluation (pycocotools), visualisation, submission |
| `cv_kfold_kaggle.ipynb` + `kaggle_kernel/` | **v1**: YOLO26m camera 5-fold CV + TTA (Kaggle GPU) |
| `cv_kfold_kaggle_v2.ipynb` + `kaggle_kernel_v2/` | **v2**: v1 + cross-camera copy-paste of rare classes + photometric variants |
| `experiments/ensemble_wbf.py` | fuses fold test predictions with WBF → final submission CSV |
| `tools/build_cv_nb_v1.py`, `tools/build_cv_nb_v2.py` | generate the two CV notebooks |
| `experiments/postproc_eval.py`, `cv_analysis.py`, `label_audit.py` | post-processing study, CV figures, label audit |
| `report/build_report.py` (+ `results.json`, `figures/`) | builds the report |
| `requirements.txt`, `README.md` | environment and how to run |

Git tag `v1-yolo26m-cv5-0.60457` marks the code state of the first YOLO26m submission.

### (2) Uploaded CSV files (`submissions/`)
| File | Model | Public |
|---|---|---|
| `yolo26s_960_tta.csv` | YOLO26s, 11 cameras (first two uploads used wrong long ids → 0.000) | 0.51648 |
| `yolo26s_960_full_tta.csv` | YOLO26s, 15 cameras | 0.54814 |
| `yolo26m_960_cv5fold_wbf.csv` | v1 5 folds, WBF 0.55 / avg | 0.60457 |
| `yolo26m_960_cv5fold_wbf_iou07.csv` | v1 5 folds, WBF 0.7 / box_and_model_avg | 0.61367 |
| `yolo26m_960_v2_5models_wbf07.csv` | v2 5 folds, WBF 0.7 | 0.60237 |
| **`yolo26m_960_v1v2_10models_wbf07.csv`** | **final: v1 + v2, 10 models, WBF 0.7** | **0.62963** |

### (3) Prepared data (Google Drive)
| File | Content |
|---|---|
| `prepared_data_v1.zip` | YOLO-format dataset of v1: cleaned labels, repeat-factor copies, fold lists `train_f{k}.txt` / `val_f{k}.txt`, `fold{k}.yaml` (relative paths) |
| `prepared_data_v2.zip` | same for v2: + 500 copy-paste images per fold (`*__cp{k}_*`), photometric repeat copies, 8 sliver boxes removed |

Both are regenerated automatically by the notebooks from the raw Kaggle data (`data/raw`, seed 42).

### (4) Model weights (Google Drive)
| File | Content |
|---|---|
| `weights_final_yolo26m_v1_v2.zip` | the 10 models of the final submission: `yolo26m_960_cv_fold{0..4}.pt` (v1), `yolo26m_960_cv_v2_fold{0..4}.pt` (v2) |
| `weights_yolo26s.zip` | YOLO26s experiment models (`yolo26s_960_full.pt`, `yolo26s_960.pt`, `yolo26s_640.pt`) |

## Part 2 — report
`report/midterm_report.docx` and `report/midterm_report.pdf` (Chapters 1–6 + appendix).

## How to reproduce the final CSV
1. `conda create -n dsde python=3.12`; `pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126`; `pip install -r requirements.txt`
2. `kaggle competitions download -c 2110531-dsde-2026-1 -p data/raw` and unzip.
3. **Without training:** copy `yolo26m_960_cv_fold{k}.pt` → `cv_out/fold{k}/best.pt` and `yolo26m_960_cv_v2_fold{k}.pt` →
   `cv_out_v2/fold{k}/best.pt`; run `cv_kfold_kaggle.ipynb` and `cv_kfold_kaggle_v2.ipynb` (they only predict when `best.pt`
   exists; `PRED_BATCH=8` on a 4 GB GPU); then
   `python experiments/ensemble_wbf.py submissions/yolo26m_960_v1v2_10models_wbf07.csv 0.7 box_and_model_avg cv_out cv_out_v2`.
   (Verified: local re-prediction matches the Kaggle fold scores within 0.0002.)
   The exact fold predictions of the submitted runs are also in `experiments/kaggle_cv_run/cv_out` and
   `experiments/kaggle_cv_run_v2/cv_out_v2`, so the CSV can be rebuilt with the same command without a GPU.
4. **With training (~9 h GPU per run):** `kaggle kernels push -p kaggle_kernel[_v2] --accelerator NvidiaTeslaT4` with the private
   datasets `chinnakrit/dsde-midterm-offline-pkgs` and `chinnakrit/dsde-midterm-data`, or run the notebooks on any GPU machine.
