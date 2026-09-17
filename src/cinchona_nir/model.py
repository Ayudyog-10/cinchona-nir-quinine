"""Scan-level neural-network screening model with qualitative banding and applicability-domain checks."""
import json
import warnings
from sklearn.exceptions import ConvergenceWarning
import numpy as np
import joblib
import sklearn
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from .preprocess import snv_sg1, align_wavelengths

DEFAULT_BANDS = {"low_below": 3.5, "high_above": 5.5, "borderline_margin": 0.5}


def _augment(Z, rng, copies=1):
    out = [Z]
    for _ in range(copies):
        out.append(Z * (1 + rng.normal(0, 0.02, (len(Z), 1))) + rng.normal(0, 0.002, (len(Z), 1)))
    return np.vstack(out)


class QuinineScreeningModel:
    def __init__(self, n_components=30, hidden=(64, 32), alpha=3.0, max_iter=150, seed=0, bands=None):
        self.n_components, self.hidden, self.alpha = n_components, hidden, alpha
        self.max_iter, self.seed = max_iter, seed
        self.bands = dict(bands or DEFAULT_BANDS)
        self.meta = {}

    # ---------- training ----------
    def fit(self, X_scans, y_scans, wavelengths):
        rng = np.random.default_rng(self.seed)
        self.wavelengths = np.asarray(wavelengths, float)
        Z = snv_sg1(X_scans)
        Za = _augment(Z, rng)
        ya = np.tile(np.asarray(y_scans, float), len(Za) // len(Z))
        self.scaler = StandardScaler().fit(Z)
        self.pca = PCA(self.n_components, random_state=self.seed).fit(self.scaler.transform(Z))
        F = self._features(Z)
        self.mlp = MLPRegressor(hidden_layer_sizes=self.hidden, alpha=self.alpha, learning_rate_init=1e-3,
                                max_iter=self.max_iter, random_state=self.seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.mlp.fit(self._features(Za), ya)
        # applicability domain: Hotelling T2 and Q residual limits (99th percentile of training scans)
        t2, q = self._t2_q(Z, F)
        self.t2_limit, self.q_limit = float(np.percentile(t2, 99)), float(np.percentile(q, 99))
        self.y_range = (float(np.min(y_scans)), float(np.max(y_scans)))
        return self

    def _features(self, Z):
        return self.pca.transform(self.scaler.transform(Z))

    def _t2_q(self, Z, F=None):
        S = self.scaler.transform(Z)
        F = self.pca.transform(S) if F is None else F
        t2 = ((F ** 2) / self.pca.explained_variance_).sum(1)
        q = ((S - self.pca.inverse_transform(F)) ** 2).sum(1)
        return t2, q

    # ---------- prediction ----------
    def band(self, value):
        lo, hi, m = self.bands["low_below"], self.bands["high_above"], self.bands["borderline_margin"]
        label = "Low" if value < lo else ("High" if value > hi else "Medium")
        borderline = min(abs(value - lo), abs(value - hi)) < m
        return label, bool(borderline)

    def predict_scans(self, X, wavelengths=None):
        wl = self.wavelengths if wavelengths is None else wavelengths
        Z = snv_sg1(align_wavelengths(X, wl, self.wavelengths))
        F = self._features(Z)
        t2, q = self._t2_q(Z, F)
        return self.mlp.predict(F), (t2 <= self.t2_limit) & (q <= self.q_limit)

    def predict_samples(self, X, sample_ids, wavelengths=None):
        """Average scan predictions per sample; returns a list of dict results."""
        pred, inside = self.predict_scans(X, wavelengths)
        sample_ids = np.asarray(sample_ids)
        out = []
        for s in dict.fromkeys(sample_ids):  # preserves order
            m = sample_ids == s
            v = float(pred[m].mean())
            label, border = self.band(v)
            frac_in = float(inside[m].mean())
            if frac_in < 0.5:
                status = "Out of model domain - confirm by HPLC"
            elif border:
                status = "Borderline - confirm by HPLC"
            else:
                status = "OK (screening result)"
            out.append({
                "sample_id": str(s), "n_scans": int(m.sum()),
                "screening_index": round(v, 2),
                "scan_spread_sd": round(float(pred[m].std()), 2),
                "class": label, "status": status,
                "in_domain_fraction": round(frac_in, 2),
            })
        return out

    # ---------- persistence ----------
    def save(self, path, meta=None):
        self.meta.update(meta or {})
        self.meta.update({"sklearn_version": sklearn.__version__, "bands": self.bands,
                          "wavelength_range_nm": [float(self.wavelengths.min()), float(self.wavelengths.max())],
                          "n_wavelengths": int(len(self.wavelengths))})
        joblib.dump(self, path)
        with open(str(path).rsplit(".", 1)[0] + "_card.json", "w") as f:
            json.dump(self.meta, f, indent=2)

    @staticmethod
    def load(path):
        return joblib.load(path)
