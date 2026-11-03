"""LLM input = CSP features as text. CSP (common spatial patterns) is fit on the 16 training
people only; each trial becomes 6 log-variance values, z-scored with training statistics.
Overwrites repro/train.jsonl and repro/test.jsonl."""
import json
import numpy as np
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

W = __file__.rsplit("/repro/", 1)[0]
SYSTEM = ("You decode motor imagery from scalp EEG recorded while a person imagined squeezing one fist for 4 s. "
          "The EEG was reduced to 6 common-spatial-pattern components (8-30 Hz). Each line is one component's "
          "log-variance as a z-score. Components 1-3 rise with one hand, 4-6 with the other. "
          "Answer with one word: LEFT or RIGHT.")

z = np.load(f"{W}/repro/arrays.npz")
tr, te = z["split"] == "train", z["split"] == "test"
X, y = z["epoch"].astype(np.float64), z["y"].astype(int)
csp = CSP(n_components=6, reg="ledoit_wolf", log=True, transform_into="average_power").fit(X[tr], y[tr])
F = csp.transform(X)
mu, sd = F[tr].mean(0), F[tr].std(0)
F = (F - mu) / sd
lab = np.where(z["y"], "RIGHT", "LEFT")
text = lambda v: "\n".join(f"component {k + 1}: {v[k]:+.2f}" for k in range(6))
for split, m in (("train", tr), ("test", te)):
    with open(f"{W}/repro/{split}.jsonl", "w") as fh:
        for i in np.where(m)[0]:
            fh.write(json.dumps({"id": str(z["ids"][i]), "messages": [
                {"role": "system", "content": SYSTEM}, {"role": "user", "content": text(F[i])},
                {"role": "assistant", "content": lab[i]}]}) + "\n")
p = LinearDiscriminantAnalysis().fit(F[tr], y[tr]).predict(F[te])
print("LDA on the same 6 numbers, test acc", round((p == y[te]).mean(), 3), f"({(p == y[te]).sum()}/{te.sum()})")
print(text(F[np.where(te)[0][0]]), "->", lab[np.where(te)[0][0]])
