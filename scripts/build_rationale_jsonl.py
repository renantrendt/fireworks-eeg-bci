"""Distill CSP+LDA into the LLM: same 6 CSP numbers as input, but the assistant answer now
shows LDA's arithmetic (weighted sum, sign rule) before the final word.

Writes repro/train_rationale.jsonl (576 rows, Fireworks format) and repro/lda_rule.json.
The test file stays repro/test.jsonl so eval_llm.py scores it unchanged.
"""
import json
import numpy as np
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

W = __file__.rsplit("/repro/", 1)[0]
SYSTEM = json.loads(open(f"{W}/repro/train.jsonl").readline())["messages"][0]["content"]

z = np.load(f"{W}/repro/arrays.npz")
tr, te = z["split"] == "train", z["split"] == "test"
X, y = z["epoch"].astype(np.float64), z["y"].astype(int)
csp = CSP(n_components=6, reg="ledoit_wolf", log=True, transform_into="average_power").fit(X[tr], y[tr])
F = csp.transform(X)
mu, sd = F[tr].mean(0), F[tr].std(0)
F = (F - mu) / sd
lda = LinearDiscriminantAnalysis().fit(F[tr], y[tr])
w, b = lda.coef_[0], float(lda.intercept_[0])
print("LDA weights", np.round(w, 2), "bias", round(b, 2))
print("LDA test acc", round((lda.predict(F[te]) == y[te]).mean(), 3))
json.dump(dict(weights=w.tolist(), bias=b, rule="score>0 -> RIGHT"), open(f"{W}/repro/lda_rule.json", "w"))

text = lambda v: "\n".join(f"component {k + 1}: {v[k]:+.2f}" for k in range(6))


def rationale(v, label):
    s = float(w @ v + b)
    terms = " ".join(f"{w[k]:+.2f}×({v[k]:+.2f})" for k in range(6))
    return (f"Weighted sum: {terms} {b:+.2f} = {s:+.2f}. "
            f"A positive score means RIGHT, negative means LEFT. Answer: {label}")


n = 0
with open(f"{W}/repro/train_rationale.jsonl", "w") as fh:
    for i in np.where(tr)[0]:
        # distillation target = LDA's own answer, so the arithmetic and the word always agree
        label = "RIGHT" if (w @ F[i] + b) > 0 else "LEFT"
        fh.write(json.dumps({"messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": text(F[i])},
            {"role": "assistant", "content": rationale(F[i], label)}]}) + "\n")
        n += 1
print(n, "rows written")
print(rationale(F[np.where(tr)[0][0]], "RIGHT" if y[np.where(tr)[0][0]] else "LEFT"))
