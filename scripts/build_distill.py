"""Distill + harness files. Teacher = LDA on the 6 CSP numbers (fit on training people).
- distill_train.jsonl: 300 balanced training trials, label = LDA's answer (not the true label).
- test_harness.jsonl: test prompts with the LDA decision score added (for the base-model harness)."""
import json
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

W = __file__.rsplit("/repro/", 1)[0]
rd = lambda f: [json.loads(l) for l in open(f"{W}/repro/{f}")]
tr, te = rd("train.jsonl"), rd("test.jsonl")
X = lambda rows: np.array([[float(l.split(":")[1]) for l in r["messages"][1]["content"].splitlines()] for r in rows])
Xtr, Xte = X(tr), X(te)
ytr = np.array([r["messages"][2]["content"] == "RIGHT" for r in tr])
lda = LinearDiscriminantAnalysis().fit(Xtr, ytr)
ptr = np.where(lda.predict(Xtr), "RIGHT", "LEFT")
rng = np.random.default_rng(0)
idx = np.concatenate([rng.choice(np.where(ptr == c)[0], 150, replace=False) for c in ("LEFT", "RIGHT")])
rng.shuffle(idx)
with open(f"{W}/repro/distill_train.jsonl", "w") as f:
    for i in idx:
        m = tr[i]["messages"]
        f.write(json.dumps({"messages": [m[0], m[1], {"role": "assistant", "content": ptr[i]}]}) + "\n")
s = lda.decision_function(Xte)
SYS = ("You decode motor imagery from scalp EEG recorded while a person imagined squeezing one fist for 4 s. "
       "You get 6 common-spatial-pattern components (log-variance z-scores) and the score of a classic linear "
       "decoder trained on other people (positive = RIGHT, negative = LEFT; larger magnitude = more confident). "
       "Answer with one word: LEFT or RIGHT.")
with open(f"{W}/repro/test_harness.jsonl", "w") as f:
    for r, v in zip(te, s):
        m = r["messages"]
        f.write(json.dumps({"id": r["id"], "messages": [{"role": "system", "content": SYS},
                 {"role": "user", "content": m[1]["content"] + f"\nlinear decoder score: {v:+.2f}"}, m[2]]}) + "\n")
print("teacher agrees with truth on train:", (ptr == np.where(ytr, "RIGHT", "LEFT")).mean().round(3), "| distill rows", len(idx))
