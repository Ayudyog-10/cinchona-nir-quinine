import numpy as np
from scipy.signal import savgol_filter


def snv(X):
    X = np.asarray(X, float)
    return (X - X.mean(1, keepdims=True)) / X.std(1, keepdims=True)


def snv_sg1(X, window=25, poly=2):
    """SNV followed by Savitzky-Golay first derivative (the tool's preprocessing)."""
    return savgol_filter(snv(X), window, poly, deriv=1, axis=1)


def align_wavelengths(X, wl_in, wl_model, tol_nm=2.0):
    """Interpolate spectra onto the model's wavelength grid. Refuses to extrapolate."""
    wl_in = np.asarray(wl_in, float)
    if len(wl_in) == len(wl_model) and np.allclose(wl_in, wl_model, atol=1e-6):
        return np.asarray(X, float)
    if wl_in.min() > wl_model.min() + tol_nm or wl_in.max() < wl_model.max() - tol_nm:
        raise ValueError(
            f"Input covers {wl_in.min():.0f}-{wl_in.max():.0f} nm; model needs "
            f"{wl_model.min():.0f}-{wl_model.max():.0f} nm."
        )
    order = np.argsort(wl_in)
    return np.array([np.interp(wl_model, wl_in[order], row[order]) for row in np.asarray(X, float)[:, order]])


def flag_outlier_scans(X, groups, z=3.5):
    """Within each group, flag scans whose SNV spectrum deviates strongly from the group median."""
    S = snv(X)
    keep = np.ones(len(X), bool)
    for g in np.unique(groups):
        i = np.flatnonzero(groups == g)
        if len(i) < 3:
            continue
        med = np.median(S[i], 0)
        d = np.sqrt(((S[i] - med) ** 2).mean(1))
        mad = np.median(np.abs(d - np.median(d))) + 1e-9
        keep[i[(d - np.median(d)) / (1.4826 * mad) > z]] = False
    return keep
