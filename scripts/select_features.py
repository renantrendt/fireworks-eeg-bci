"""Choose the text feature set by leave-one-person-out CV on the 16 TRAINING people only.
The test people are not touched here."""
import numpy as np
from scipy.signal import welch
from mne.decoding import CSP
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import mne

W = __file__.rsplit("/repro/", 1)[0]
z = np.load(f"{W}/repro/arrays.npz")
tr = z["split"] == "train"
X, y, P = z["epoch"][tr].astype(np.float64), z["y"][tr].astype(int), z["person"][tr]
info = mne.io.read_info(f"{W}/data/info-info.fif", verbose=0)
ch = info.ch_names
idx = lambda n: ch.index(n)
f, psd = welch(X, fs=160, nperseg=160, axis=-1)
band = lambda lo, hi: np.log(psd[..., (f >= lo) & (f <= hi)].mean(-1))
mu, beta = band(8, 12), band(13, 30)
motor = [idx(c) for c in ["FC3", "FC4", "C3", "Cz", "C4", "CP3", "CP4", "C5", "C6", "C1", "C2"]]


def lat(b):  # left-minus-right hemisphere log power, a person-free quantity
    pairs = [("C3", "C4"), ("FC3", "FC4"), ("CP3", "CP4"), ("C5", "C6"), ("C1", "C2")]
    return np.stack([b[:, idx(a)] - b[:, idx(c)] for a, c in pairs], 1)


cands = {
    "logpow_motor": np.concatenate([mu[:, motor], beta[:, motor]], 1),
    "laterality_mu_beta": np.concatenate([lat(mu), lat(beta)], 1),
    "laterality_mu_beta_centered": None,  # trial log power minus that person's mean over all trials (no labels)
}
c = np.concatenate([lat(mu), lat(beta)], 1)
cands["laterality_mu_beta_centered"] = c - np.stack([c[P == p].mean(0) for p in P])


def lopo(make, F):
    accs = []
    for p in np.unique(P):
        m = make(); m.fit(F[P != p], y[P != p]); accs.append((m.predict(F[P == p]) == y[P == p]).mean())
    return np.mean(accs)


lr = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
for k, F in cands.items():
    print(f"{k:30s} LOPO acc {lopo(lr, F):.3f}")
print(f"{'csp_lda (raw epochs)':30s} LOPO acc {lopo(lambda: make_pipeline(CSP(6, reg='ledoit_wolf', log=True), LinearDiscriminantAnalysis()), X):.3f}")
