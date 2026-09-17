"""Streamlit web app: Cinchona quinine NIR screening (qualitative). Run: streamlit run app/app.py"""
import sys, json, tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cinchona_nir.data import read_spectra_file
from cinchona_nir.model import QuinineScreeningModel

MODEL_PATH = ROOT / "models" / "quinine_screening_model.joblib"
CARD_PATH = ROOT / "models" / "quinine_screening_model_card.json"

st.set_page_config(page_title="Cinchona Quinine NIR Screening", page_icon="🌿", layout="wide")


@st.cache_resource
def load_model():
    return QuinineScreeningModel.load(MODEL_PATH)


model = load_model()
card = json.loads(CARD_PATH.read_text()) if CARD_PATH.exists() else {}

st.title("🌿 Cinchona bark – quinine NIR screening")
st.warning(
    "**Qualitative screening tool.** Results sort samples into Low / Medium / High groups to prioritise "
    "laboratory work. They are **not quantitative** and **do not replace HPLC**. Borderline and "
    "out-of-domain samples must be confirmed by HPLC."
)

with st.sidebar:
    st.header("Model")
    b = model.bands
    st.markdown(
        f"- **Low:** index < {b['low_below']}\n- **Medium:** {b['low_below']}–{b['high_above']}\n"
        f"- **High:** index > {b['high_above']}\n- **Borderline:** within ±{b['borderline_margin']} of a cut-off"
    )
    lo, hi = card.get("wavelength_range_nm", [None, None])
    if lo:
        st.markdown(f"**Wavelength range:** {lo:.0f}–{hi:.0f} nm ({card.get('n_wavelengths')} points)")
    cv = card.get("cv_10fold")
    if cv:
        c = cv["classification"]
        st.markdown(
            f"**Cross-validated performance** ({card.get('n_samples')} samples)\n\n"
            f"- Class accuracy: {c['accuracy']:.0%} (balanced {c['balanced_accuracy']:.0%})\n"
            f"- Accuracy when not borderline: {c['accuracy_non_borderline']:.0%}\n"
            f"- Low↔High confusion: {c['gross_error_rate(Low<->High)']:.1%}\n"
            f"- R² {cv['R2']:.2f}, RPD {cv['RPD']:.2f}"
        )
    st.caption(f"Trained {card.get('trained_at', '?')} on {card.get('trained_on', '?')}")

st.subheader("1. Upload spectra")
st.markdown(
    "CSV or Excel, **one scan per row**, wavelength values (nm) as column headers. "
    "Add a `sample_id` column so replicate scans of the same sample are averaged."
)
example = ROOT / "data" / "example_scans.csv"
if example.exists():
    st.download_button("Download example file", example.read_bytes(), "example_scans.csv", "text/csv")
up = st.file_uploader("Spectra file", type=["csv", "xlsx", "xls"])

if up is not None:
    tmp = Path(tempfile.mkdtemp()) / up.name
    tmp.write_bytes(up.getvalue())
    try:
        X, wl, ids = read_spectra_file(tmp)
        res = pd.DataFrame(model.predict_samples(X, ids, wl))
    except Exception as e:
        st.error(f"Could not process file: {e}")
        st.stop()

    st.subheader("2. Results")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Samples", len(res))
    c2.metric("Low", int((res["class"] == "Low").sum()))
    c3.metric("Medium", int((res["class"] == "Medium").sum()))
    c4.metric("High", int((res["class"] == "High").sum()))
    n_conf = int((res["status"] != "OK (screening result)").sum())
    if n_conf:
        st.info(f"{n_conf} sample(s) need HPLC confirmation (borderline or outside model domain).")

    colour = {"Low": "#f6d5c8", "Medium": "#f3ecd0", "High": "#cfe6d8"}
    st.dataframe(res.style.apply(lambda r: [f"background-color: {colour[r['class']]}"] * len(r), axis=1),
                 use_container_width=True)
    chart = res.set_index("sample_id")[["screening_index"]]
    st.bar_chart(chart)
    st.download_button("Download results (CSV)", res.to_csv(index=False).encode(), "screening_results.csv", "text/csv")
    st.caption("Screening index is a relative indicator on the HPLC reference scale; it is not a reportable quinine content.")
