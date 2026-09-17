"""Train the final screening model and (optionally) estimate its performance by 10-fold CV.

Usage:
    python scripts/train.py --data data/whole_Cinchona_Combine_Data_wih_Reference.xlsx --cv
"""
import argparse, json, sys, datetime
from pathlib import Path
import numpy as np
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cinchona_nir.data import load_workbook, CAMPAIGNS
from cinchona_nir.preprocess import flag_outlier_scans
from cinchona_nir.model import QuinineScreeningModel


def class_metrics(t, p, model):
    tc = np.array([model.band(v)[0] for v in t]); pc = np.array([model.band(v)[0] for v in p])
    labels = ["Low", "Medium", "High"]
    cm = [[int(((tc == a) & (pc == b)).sum()) for b in labels] for a in labels]
    recall = {a: float(((tc == a) & (pc == a)).sum() / max((tc == a).sum(), 1)) for a in labels}
    gross = float(((tc == "Low") & (pc == "High")).sum() + ((tc == "High") & (pc == "Low")).sum()) / len(t)
    border = np.array([model.band(v)[1] for v in p])
    return {"labels": labels, "confusion_matrix(rows=true)": cm, "accuracy": float((tc == pc).mean()),
            "balanced_accuracy": float(np.mean(list(recall.values()))), "recall": recall,
            "gross_error_rate(Low<->High)": gross,
            "accuracy_non_borderline": float((tc[~border] == pc[~border]).mean()),
            "borderline_fraction": float(border.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="models/quinine_screening_model.joblib")
    ap.add_argument("--cv", action="store_true", help="run 10-fold sample-level CV first")
    a = ap.parse_args()

    X, y, wl, sid, camp = load_workbook(a.data)
    keep = flag_outlier_scans(X, sid)
    X, y, sid, camp = X[keep], y[keep], sid[keep], camp[keep]
    print(f"{len(X)} scans kept ({(~keep).sum()} outlier scans removed), {len(np.unique(sid))} samples")
    meta = {"trained_on": a.data.split("/")[-1], "trained_at": datetime.date.today().isoformat(),
            "n_samples": int(len(np.unique(sid))), "n_scans": int(len(X)), "campaigns": [c[0] for c in CAMPAIGNS],
            "intended_use": "Qualitative screening (Low / Medium / High). Does NOT replace HPLC quantification."}

    if a.cv:
        ids = np.unique(sid); t = np.array([y[sid == s][0] for s in ids]); P = np.zeros(len(ids))
        for k, (tr, te) in enumerate(KFold(10, shuffle=True, random_state=0).split(ids)):
            mtr = np.isin(sid, ids[tr]); mte = np.isin(sid, ids[te])
            m = QuinineScreeningModel().fit(X[mtr], y[mtr], wl)
            res = m.predict_samples(X[mte], sid[mte])
            P[te] = [r["screening_index"] for r in res]
            print(f"  fold {k+1}/10 done")
        rmse = float(np.sqrt(np.mean((P - t) ** 2)))
        meta["cv_10fold"] = {"R2": float(1 - np.sum((P - t) ** 2) / np.sum((t - t.mean()) ** 2)),
                             "RMSECV": rmse, "RPD": float(t.std() / rmse),
                             "classification": class_metrics(t, P, m)}
        Path("reports").mkdir(exist_ok=True)
        np.savetxt("reports/cv_predictions.csv", np.c_[ids, t, P], delimiter=",",
                   header="sample,reference,predicted", comments="", fmt=["%d", "%.2f", "%.3f"])
        print(json.dumps(meta["cv_10fold"], indent=2))

    model = QuinineScreeningModel().fit(X, y, wl)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    model.save(a.out, meta)
    print("saved", a.out)


if __name__ == "__main__":
    main()
