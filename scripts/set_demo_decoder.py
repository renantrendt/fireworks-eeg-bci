"""Put a real decoder's per-trial predictions into data/trials.npz (the file the demo videos read),
replacing the mock guesses. Usage: set_demo_decoder.py csp_lda | glm53_lora"""
import json, sys
import numpy as np

W = __file__.rsplit("/repro/", 1)[0]
which = sys.argv[1]
pred = {}
for line in open(f"{W}/repro/predictions.csv").read().splitlines()[1:]:
    cols = line.split(",")
    pred[cols[0]] = cols
hdr = open(f"{W}/repro/predictions.csv").readline().strip().split(",")
col = hdr.index(which)
conf = {}
if which == "csp_lda":
    from mne.decoding import CSP
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.pipeline import make_pipeline
    z = np.load(f"{W}/repro/arrays.npz")
    tr, te = z["split"] == "train", z["split"] == "test"
    m = make_pipeline(CSP(6, reg="ledoit_wolf", log=True), LinearDiscriminantAnalysis()).fit(z["epoch"][tr].astype(float), z["y"][tr])
    pr = m.predict_proba(z["epoch"][te].astype(float)).max(1)
    conf = dict(zip(z["ids"][te], pr))

d = dict(np.load(f"{W}/data/trials.npz", allow_pickle=True))
meta = json.loads(str(d["meta"]))
names = list(d["run_names"])
k_in_run = {}
for t in meta:
    k = k_in_run.get(t["run"], 0); k_in_run[t["run"]] = k + 1
    tid = f"{names[t['run']]}_{k}"
    row = pred[tid]
    assert row[1][0] == t["truth"], tid
    t["guess"] = row[col][0]
    t["conf"] = float(conf.get(tid, 0.0))
    t["decoder"] = which
d["meta"] = json.dumps(meta)
np.savez_compressed(f"{W}/data/trials.npz", **d)
print(which, "acc", np.mean([t["truth"] == t["guess"] for t in meta]), "on", len(meta))
