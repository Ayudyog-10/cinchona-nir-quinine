from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cinchona_nir.data import read_spectra_file
from cinchona_nir.model import QuinineScreeningModel


def test_example_prediction():
    m = QuinineScreeningModel.load(ROOT / "models" / "quinine_screening_model.joblib")
    X, wl, ids = read_spectra_file(ROOT / "data" / "example_scans.csv")
    res = m.predict_samples(X, ids, wl)
    assert len(res) == 3
    assert all(r["class"] in {"Low", "Medium", "High"} for r in res)
    assert all(r["in_domain_fraction"] > 0.5 for r in res)
