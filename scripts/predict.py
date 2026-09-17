"""Predict screening class for new spectra.

Usage:
    python scripts/predict.py new_scans.csv --id-column sample_id --out results.csv
Input: one scan per row, wavelength (nm) column headers; a sample-id column groups replicate scans.
"""
import argparse, sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cinchona_nir.data import read_spectra_file
from cinchona_nir.model import QuinineScreeningModel

ap = argparse.ArgumentParser()
ap.add_argument("spectra")
ap.add_argument("--model", default="models/quinine_screening_model.joblib")
ap.add_argument("--out", default="predictions.csv")
a = ap.parse_args()
m = QuinineScreeningModel.load(a.model)
X, wl, ids = read_spectra_file(a.spectra)
df = pd.DataFrame(m.predict_samples(X, ids, wl))
df.to_csv(a.out, index=False)
print(df.to_string(index=False))
print("\nScreening results only - confirm decisions by HPLC.")
