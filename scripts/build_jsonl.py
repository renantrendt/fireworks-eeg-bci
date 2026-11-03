"""Write the LLM train/test files from the feature set chosen by select_features.py:
left-minus-right hemisphere power (dB) in mu and beta for 5 electrode pairs."""
import json
import numpy as np, mne
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

W = __file__.rsplit("/repro/", 1)[0]
PAIRS = [("C3", "C4"), ("FC3", "FC4"), ("CP3", "CP4"), ("C5", "C6"), ("C1", "C2")]
SYSTEM = ("You decode motor imagery from scalp EEG recorded while a person imagined squeezing one fist for 4 s. "
          "Each line is left-hemisphere minus right-hemisphere band power in dB for one electrode pair. "
          "Answer with one word: LEFT or RIGHT.")

z = np.load(f"{W}/repro/arrays.npz")
ch = mne.io.read_info(f"{W}/data/info-info.fif", verbose=0).ch_names
f, psd = welch(z["epoch"].astype(np.float64), fs=160, nperseg=160, axis=-1)
db = lambda lo, hi: 10 * np.log10(psd[..., (f >= lo) & (f <= hi)].mean(-1))
mu, beta = db(8, 12), db(13, 30)
F = np.stack([b[:, ch.index(a)] - b[:, ch.index(c)] for b in (mu, beta) for a, c in PAIRS], 1)
lab = np.where(z["y"], "RIGHT", "LEFT")


def text(v):
    return "\n".join(f"{a}-{c}: mu {v[i]:+.2f} dB, beta {v[i + 5]:+.2f} dB" for i, (a, c) in enumerate(PAIRS))


for split in ("train", "test"):
    with open(f"{W}/repro/{split}.jsonl", "w") as fh:
        for i in np.where(z["split"] == split)[0]:
            fh.write(json.dumps({"id": str(z["ids"][i]), "messages": [
                {"role": "system", "content": SYSTEM}, {"role": "user", "content": text(F[i])},
                {"role": "assistant", "content": lab[i]}]}) + "\n")
tr, te = z["split"] == "train", z["split"] == "test"
m = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)).fit(F[tr], z["y"][tr])
p = m.predict(F[te])
r = json.load(open(f"{W}/repro/classic_results.json"))
r["logreg_text_features"] = dict(acc=float((p == z["y"][te]).mean()), pred=p.astype(int).tolist())
json.dump(r, open(f"{W}/repro/classic_results.json", "w"))
print("logreg on the LLM's numbers, test acc", round(r["logreg_text_features"]["acc"], 3))
print(text(F[0]), "->", lab[0])
