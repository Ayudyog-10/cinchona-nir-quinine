# Cinchona NIR Quinine Screening (METASPEQ)

Near-infrared (NIR, 892–1710 nm) screening of quinine in *Cinchona* bark from Mangpoo, Darjeeling.
The model sorts samples into **Low / Medium / High** groups to prioritise laboratory work.

> ⚠️ **Intended use: qualitative screening only.** The model does **not** replace HPLC.
> Its cross-validated error (RMSECV ≈ 1.0, RPD ≈ 1.55) is too large for quantitative reporting.
> Borderline and out-of-domain results must be confirmed by HPLC.

## Repository layout

```
src/cinchona_nir/   data loading, preprocessing (SNV + Savitzky–Golay 1st derivative), model class
scripts/train.py      train final model (+ --cv for 10-fold cross-validated performance)
scripts/benchmark.py  compare 11 modelling methods with nested CV (white-paper tables)
scripts/predict.py    command-line prediction for new spectra
app/app.py            Streamlit web app
models/               quinine_screening_model.joblib + model card (performance, bands, metadata)
data/                 example input; place the full workbook here to retrain (not committed)
tests/                smoke test
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/app.py                               # opens http://localhost:8501
```

Command line:

```bash
python scripts/predict.py data/example_scans.csv --out predictions.csv
```

Input format: one scan per row, wavelength (nm) numbers as column headers, plus a `sample_id`
column. Replicate scans with the same `sample_id` are averaged. Spectra on a slightly different
wavelength grid are interpolated; spectra that do not cover 892–1710 nm are rejected.

## Model

| Item | Setting |
|---|---|
| Preprocessing | outlier-scan removal (robust z > 3.5) → SNV → Savitzky–Golay 1st derivative (window 25, order 2) |
| Features | standardisation → PCA (30 components) |
| Regressor | neural network (MLP 64-32, L2 α = 3) trained on individual scans + jittered copies |
| Output | mean over a sample's scans → screening index → class |
| Classes | Low < 3.5 ≤ Medium ≤ 5.5 < High; "borderline" within ±0.5 of a cut-off |
| Domain check | Hotelling T² and Q-residual limits (99th percentile of training scans) |

### Cross-validated performance (10-fold, 268 samples)

| Metric | Value |
|---|---|
| R² / RMSECV / RPD | 0.58 / 1.00 / 1.55 |
| 3-class accuracy (balanced) | 74% (69%) — majority-class guess = 52% |
| Accuracy on non-borderline samples | 81% (62% of samples) |
| Low ↔ High confusion | 1.1% |
| Recall Low / Medium / High | 65% / 86% / 58% |

The class cut-offs (3.5 / 5.5) are working values on the HPLC reference scale; the units of the
reference and final cut-offs should be confirmed by the laboratory before operational use.

## Retraining and benchmarking

```bash
cp /path/to/whole_Cinchona_Combine_Data_wih_Reference.xlsx data/
python scripts/train.py --data data/whole_Cinchona_Combine_Data_wih_Reference.xlsx --cv
python scripts/benchmark.py --data data/whole_Cinchona_Combine_Data_wih_Reference.xlsx   # ~20–40 min
```

The model file is tied to **scikit-learn 1.8.0** (pinned in `requirements.txt`). If you upgrade
scikit-learn, retrain the model.

## Deployment

- **Streamlit Community Cloud:** connect the GitHub repo, main file `app/app.py`.
- **Docker:** `docker build -t cinchona-nir . && docker run -p 8501:8501 cinchona-nir`

## Known limitations

- HPLC reference accuracy is unverified; this caps achievable model accuracy.
- Models trained on one campaign do not transfer to another. New instruments, operators or sample
  preparation require re-validation (and ideally calibration transfer).
- No independent external test set yet; all performance figures are cross-validated.
