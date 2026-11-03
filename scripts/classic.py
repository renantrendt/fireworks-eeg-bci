"""Classic baselines on the same person split: CSP+LDA on raw epochs, and logistic regression
on the same 14 numbers the LLM reads (how much signal the text features carry)."""
import json
import numpy as np
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

W = __file__.rsplit("/repro/", 1)[0]
z = np.load(f"{W}/repro/arrays.npz")
tr, te = z["split"] == "train", z["split"] == "test"
y = z["y"].astype(int)
res = {}
models = {
    "csp_lda": (make_pipeline(CSP(n_components=6, reg="ledoit_wolf", log=True), LinearDiscriminantAnalysis()), z["epoch"].astype(np.float64)),
    "logreg_text_features": (make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)), np.clip(z["feat"], -200, 400)),
}
for name, (m, X) in models.items():
    m.fit(X[tr], y[tr])
    p = m.predict(X[te])
    res[name] = dict(acc=float((p == y[te]).mean()), pred=p.tolist())
    print(f"{name:22s} test acc {res[name]['acc']:.3f}  ({(p == y[te]).sum()}/{te.sum()})")
json.dump(dict(ids=z["ids"][te].tolist(), y=y[te].tolist(), **res), open(f"{W}/repro/classic_results.json", "w"))
