"""Benchmark of modelling methods with nested 10-fold sample-level CV (reproduces the white-paper tables).

Usage:
    python scripts/benchmark.py --data data/whole_Cinchona_Combine_Data_wih_Reference.xlsx [--methods "Gaussian process" ...]
Runtime: roughly 20-40 minutes on one CPU for all methods.
"""
import numpy as np, pandas as pd, warnings, json, sys, argparse
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cinchona_nir.data import load_workbook
from cinchona_nir.preprocess import flag_outlier_scans
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as Ck
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--methods", nargs="*")
ap.add_argument("--out", default="reports/benchmark.csv")
args = ap.parse_args()
X, y, wl, sid, batch = load_workbook(args.data)
keep = flag_outlier_scans(X, sid)
Xk, sk = X[keep], sid[keep]
ids = np.unique(sid); M = np.array([Xk[sk == s].mean(0) for s in ids])
t = np.array([y[sid == s][0] for s in ids]); b = np.array([batch[sid == s][0] for s in ids])
snv=lambda A:(A-A.mean(1,keepdims=True))/A.std(1,keepdims=True)
def msc(A,ref):
    out=np.empty_like(A)
    for i,a in enumerate(A): c=np.polyfit(ref,a,1); out[i]=(a-c[1])/c[0]
    return out
PRE={'SNV':snv,'SNV+SG1(11)':lambda A:savgol_filter(snv(A),11,2,deriv=1,axis=1),'SNV+SG1(25)':lambda A:savgol_filter(snv(A),25,2,deriv=1,axis=1),
 'SNV+SG2(25)':lambda A:savgol_filter(snv(A),25,3,deriv=2,axis=1),'SG1(25)+SNV':lambda A:snv(savgol_filter(A,25,2,deriv=1,axis=1)),
 'SNV+detrend':lambda A:snv(A)-np.polyval(np.polyfit(np.arange(A.shape[1]),snv(A).T,2),np.arange(A.shape[1])[:,None]).T}
def rmse(a,b_):return np.sqrt(np.mean((a-b_)**2))
def plscv(A,t,folds=5,maxlv=15,seed=1):
    kf=KFold(folds,shuffle=True,random_state=seed); P=np.zeros((maxlv,len(t)))
    for tr,te in kf.split(A):
        for k in range(1,maxlv+1): P[k-1,te]=PLSRegression(k,scale=False).fit(A[tr],t[tr]).predict(A[te]).ravel()
    e=[rmse(p,t) for p in P]; return int(np.argmin(e))+1,min(e)
def intervals(n,w=16): return [np.arange(i,min(i+w,n)) for i in range(0,n,w)]
# ---- model definitions: fit(Atr,ttr,btr) -> predict(Ate,bte). Selection is done INSIDE training folds (nested).
class PLSauto:
    def __init__(s,pre): s.pre=pre
    def fit(s,A,t,b):
        s.f=PRE[s.pre]; Z=s.f(A); s.k,_=plscv(Z,t); s.m=PLSRegression(s.k,scale=False).fit(Z,t); return s
    def predict(s,A,b): return s.m.predict(s.f(A)).ravel()
class PrePLS:  # choose preprocessing + LV inside training fold
    def fit(s,A,t,b):
        best=min(((plscv(f(A),t),n) for n,f in PRE.items()),key=lambda z:z[0][1])
        s.f=PRE[best[1]]; s.m=PLSRegression(best[0][0],scale=False).fit(s.f(A),t); s.name=best[1]; return s
    def predict(s,A,b): return s.m.predict(s.f(A)).ravel()
class iPLS:  # forward interval selection, inner CV
    def fit(s,A,t,b):
        Z=snv(A); iv=intervals(Z.shape[1]); sel=[]; best=np.inf
        for _ in range(6):
            cand=[(plscv(Z[:,np.concatenate(sel+[v])],t,maxlv=10)[1],j) for j,v in enumerate(iv) if not any(v is u for u in sel)]
            e,j=min(cand)
            if e>best-0.005: break
            best=e; sel.append(iv[j])
        s.cols=np.concatenate(sel); s.k,_=plscv(Z[:,s.cols],t,maxlv=10); s.m=PLSRegression(s.k,scale=False).fit(Z[:,s.cols],t); return s
    def predict(s,A,b): return s.m.predict(snv(A)[:,s.cols]).ravel()
class VIPPLS:
    def fit(s,A,t,b):
        Z=snv(A); k,_=plscv(Z,t); m=PLSRegression(k,scale=False).fit(Z,t)
        T_,W,Q=m.x_scores_,m.x_weights_,m.y_loadings_; ss=(T_**2).sum(0)*(Q.ravel()**2)
        vip=np.sqrt(Z.shape[1]*((W/np.linalg.norm(W,axis=0))**2@ss)/ss.sum())
        best=None
        for th in [0.6,0.8,1.0,1.2]:
            c=vip>=th
            if c.sum()<10: continue
            kk,e=plscv(Z[:,c],t)
            if best is None or e<best[0]: best=(e,c,kk)
        s.c=best[1]; s.m=PLSRegression(best[2],scale=False).fit(Z[:,s.c],t); return s
    def predict(s,A,b): return s.m.predict(snv(A)[:,s.c]).ravel()
class LWPLS:  # locally weighted / local PLS
    def __init__(s,k=60,lv=6): s.k,s.lv=k,lv
    def fit(s,A,t,b):
        s.Z=snv(A); s.t=t; s.p=PCA(10).fit(s.Z); s.S=s.p.transform(s.Z); return s
    def predict(s,A,b):
        Z=snv(A); S=s.p.transform(Z); out=[]
        for z,q in zip(Z,S):
            d=np.linalg.norm(s.S-q,axis=1); nn=np.argsort(d)[:s.k]
            out.append(PLSRegression(s.lv,scale=False).fit(s.Z[nn],s.t[nn]).predict(z[None]).ravel()[0])
        return np.array(out)
class GPR:
    def fit(s,A,t,b):
        Z=savgol_filter(snv(A),25,2,deriv=1,axis=1); s.sc=StandardScaler().fit(Z); s.p=PCA(15).fit(s.sc.transform(Z))
        k=Ck(1.0)*RBF(np.ones(15)*5)+WhiteKernel(0.5)
        s.mu=t.mean(); s.m=GaussianProcessRegressor(k,normalize_y=True,n_restarts_optimizer=1,random_state=0).fit(s.p.transform(s.sc.transform(Z)),t); return s
    def predict(s,A,b):
        Z=savgol_filter(snv(A),25,2,deriv=1,axis=1); return s.m.predict(s.p.transform(s.sc.transform(Z)))
class HGB:
    def fit(s,A,t,b):
        Z=snv(A); s.p=PCA(20).fit(Z); s.m=HistGradientBoostingRegressor(max_iter=300,learning_rate=0.05,max_leaf_nodes=8,min_samples_leaf=8,l2_regularization=1.0,random_state=0).fit(s.p.transform(Z),t); return s
    def predict(s,A,b): return s.m.predict(s.p.transform(snv(A)))
class CampaignPLS:  # campaign-aware: centre spectra per campaign + campaign dummies
    def fit(s,A,t,b):
        Z=snv(A); s.cm={k:Z[b==k].mean(0) for k in np.unique(b)}
        Zc=Z-np.array([s.cm[k] for k in b]); s.k,_=plscv(Zc,t-0) ; s.pls=PLSRegression(s.k,scale=False).fit(Zc,t)
        r=t-s.pls.predict(Zc).ravel(); s.off={k:r[b==k].mean() for k in np.unique(b)}; return s
    def predict(s,A,b):
        Z=snv(A); Zc=Z-np.array([s.cm.get(k,0) for k in b]); return s.pls.predict(Zc).ravel()+np.array([s.off.get(k,0) for k in b])
class Stack:
    def fit(s,A,t,b):
        base=lambda:[PLSauto('SNV'),PLSauto('SNV+SG1(25)'),HGB(),SVRm()]
        kf=KFold(5,shuffle=True,random_state=3); O=np.zeros((len(t),4))
        for tr,te in kf.split(A):
            for j,m in enumerate(base()): O[te,j]=m.fit(A[tr],t[tr],b[tr]).predict(A[te],b[te])
        from scipy.optimize import nnls
        w,_=nnls(O,t); s.w=w/w.sum(); s.ms=[m.fit(A,t,b) for m in base()]; return s
    def predict(s,A,b): return np.column_stack([m.predict(A,b) for m in s.ms])@s.w
class SVRm:
    def fit(s,A,t,b):
        Z=savgol_filter(snv(A),25,2,deriv=1,axis=1); s.m=make_pipeline(StandardScaler(),PCA(15),SVR(C=5,epsilon=0.3,gamma='scale')).fit(Z,t); return s
    def predict(s,A,b): return s.m.predict(savgol_filter(snv(A),25,2,deriv=1,axis=1))
class ScanMLP:  # neural net trained on individual scans (augmentation), prediction = mean over a sample's scans
    def fit(s,A,t,b,ids_tr=None):
        m=np.isin(sk,s.ids_tr); Z=savgol_filter(snv(Xk[m]),25,2,deriv=1,axis=1); ty=np.array([dict(zip(ids,t_all))[q] for q in sk[m]])
        # augmentation: small baseline/scale jitter
        rng=np.random.default_rng(0); Za=Z*(1+rng.normal(0,.02,(len(Z),1)))+rng.normal(0,.002,(len(Z),1))
        s.sc=StandardScaler().fit(Z); s.p=PCA(30).fit(s.sc.transform(Z))
        F=lambda W:s.p.transform(s.sc.transform(W))
        s.m=MLPRegressor(hidden_layer_sizes=(64,32),alpha=3.0,learning_rate_init=1e-3,max_iter=150,early_stopping=False,random_state=0).fit(np.vstack([F(Z),F(Za)]),np.r_[ty,ty]); s.F=F; return s
    def predict(s,A,b):
        out=[]
        for q in s.ids_te:
            Z=savgol_filter(snv(Xk[sk==q]),25,2,deriv=1,axis=1); out.append(s.m.predict(s.F(Z)).mean())
        return np.array(out)
t_all = t
MODELS = {"PLS baseline (SNV)": lambda: PLSauto("SNV"), "PLS + auto-preprocessing": PrePLS,
          "iPLS (interval selection)": iPLS, "VIP-PLS (variable selection)": VIPPLS,
          "Local PLS (LW-PLS)": LWPLS, "Gaussian process": GPR, "Gradient boosting": HGB,
          "SVR (SG1)": SVRm, "Campaign-aware PLS": CampaignPLS, "Stacked ensemble": Stack,
          "Scan-level neural net": ScanMLP}
scen = {"All campaigns": np.isin(b, [0, 1, 2]), "B+C (excl. campaign A)": np.isin(b, [1, 2])}
rows = []
for name in (args.methods or list(MODELS)):
    for sc, msk in scen.items():
        A, tt, bb, ii = M[msk], t[msk], b[msk], ids[msk]; P = np.zeros(len(tt))
        for tr, te in KFold(10, shuffle=True, random_state=0).split(A):
            m = MODELS[name](); m.ids_tr = ii[tr]; m.ids_te = ii[te]
            P[te] = m.fit(A[tr], tt[tr], bb[tr]).predict(A[te], bb[te])
        r = rmse(P, tt)
        rows.append(dict(method=name, scenario=sc, R2=1 - np.sum((P - tt) ** 2) / np.sum((tt - tt.mean()) ** 2),
                         RMSECV=r, RPD=tt.std() / r))
        print(rows[-1], flush=True)
Path(args.out).parent.mkdir(exist_ok=True)
pd.DataFrame(rows).to_csv(args.out, index=False)
print("saved", args.out)
