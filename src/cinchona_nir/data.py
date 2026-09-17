"""Loading the combined Cinchona workbook and labelling samples/campaigns by row position."""
import numpy as np
import pandas as pd

# Row layout of 'whole_Cinchona_Combine_Data_wih_Reference.xlsx' (Excel rows 2..4265).
CAMPAIGNS = [  # (name, n_samples, scans_per_sample)
    ("A_2022-23", 6, 12),
    ("B_2022-23", 111, 16),
    ("C_2024-09", 151, 16),
]


def load_workbook(path):
    """Return X (scans x wavelengths), y (reference per scan), wavelengths, sample_id, campaign index."""
    df = pd.read_excel(path)
    y = df.iloc[:, 0].to_numpy(float)
    X = df.iloc[:, 1:].to_numpy(float)
    wl = df.columns[1:].astype(float).to_numpy()
    sid, camp = [], []
    offset = 0
    for c, (_, n, r) in enumerate(CAMPAIGNS):
        sid += list(offset + np.repeat(np.arange(n), r))
        camp += [c] * (n * r)
        offset += n
    if len(sid) != len(X):
        raise ValueError(f"Workbook has {len(X)} rows; expected {len(sid)} from CAMPAIGNS layout.")
    return X, y, wl, np.array(sid), np.array(camp)


def read_spectra_file(path):
    """Read new spectra for prediction (CSV or Excel).

    Expected layout: one scan per row; wavelength numbers as column headers.
    Optional non-numeric columns (e.g. 'sample_id') are kept as identifiers.
    An optional first column named like 'Reference...' is ignored.
    """
    df = pd.read_csv(path) if str(path).lower().endswith(".csv") else pd.read_excel(path)
    wl_cols, id_cols = [], []
    for c in df.columns:
        try:
            float(c)
            wl_cols.append(c)
        except (TypeError, ValueError):
            if not str(c).lower().startswith("reference"):
                id_cols.append(c)
    if not wl_cols:
        raise ValueError("No wavelength columns found (column headers must be numbers in nm).")
    wl = np.array([float(c) for c in wl_cols])
    X = df[wl_cols].to_numpy(float)
    ids = df[id_cols[0]].astype(str).to_numpy() if id_cols else np.array([f"row_{i+2}" for i in range(len(df))])
    return X, wl, ids
