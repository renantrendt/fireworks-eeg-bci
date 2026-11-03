"""Cut real imagery trials (left/right fist) from the held-out PhysioNet people.

Output: data/trials.npz with continuous C3/Cz/C4 signal per run, trial onsets,
true labels, per-trial mu-band power on all 64 channels (for the scalp map),
and MOCK decoder guesses at a fixed 68% hit rate. The mock guesses are
replaced by the real fine-tuned model's output later.
"""
import glob, json
import numpy as np, mne

W = __file__.rsplit("/demo/", 1)[0]
rng = np.random.default_rng(33)
runs, trials = [], []
for f in sorted(glob.glob(f"{W}/data/eeg/*.edf")):
    raw = mne.io.read_raw_edf(f, preload=True, verbose=0)
    mne.datasets.eegbci.standardize(raw)
    raw.set_montage("standard_1005")
    raw.filter(1., 40., verbose=0)
    sf = raw.info["sfreq"]
    disp = raw.get_data(picks=["C3", "Cz", "C4"]) * 1e6  # microvolts
    mu = raw.copy().filter(8., 12., verbose=0).get_data()
    rid = len(runs)
    runs.append(dict(name=f.rsplit("/", 1)[1][:-4], disp=disp))
    for on, dur, desc in zip(raw.annotations.onset, raw.annotations.duration, raw.annotations.description):
        if desc not in ("T1", "T2") or on > 100:
            continue
        a, b = int(on * sf), int((on + 4.0) * sf)
        rest = slice(max(0, a - int(2 * sf)), a)
        p_task = (mu[:, a:b] ** 2).mean(1)
        p_rest = (mu[:, rest] ** 2).mean(1) + 1e-20
        erd = 100 * (p_task - p_rest) / p_rest  # % power change vs rest
        truth = "L" if desc == "T1" else "R"
        ok = rng.random() < 0.68
        guess = truth if ok else ("R" if truth == "L" else "L")
        trials.append(dict(run=rid, onset=float(on), truth=truth, guess=guess,
                           conf=float(np.clip(rng.normal(0.78 if ok else 0.6, 0.08), 0.5, 0.99)),
                           erd=erd.astype(np.float32)))

info = raw.copy().pick("eeg").info
np.savez_compressed(f"{W}/data/trials.npz",
                    disp=np.stack([r["disp"][:, :16960] for r in runs]).astype(np.float32),
                    run_names=[r["name"] for r in runs],
                    meta=json.dumps([{k: v for k, v in t.items() if k != "erd"} for t in trials]),
                    erd=np.stack([t["erd"] for t in trials]), sfreq=sf)
mne.io.write_info(f"{W}/data/info-info.fif", info, overwrite=True)
acc = np.mean([t["truth"] == t["guess"] for t in trials])
print(len(runs), "runs", len(trials), "trials", "mock acc", round(acc, 3))
