"""Turn each imagery trial into text (EEG-GPT style) and into arrays for the classic baseline.

Split by person: S001-S016 train, S101-S104 test (the same 144 trials the demo replays).
Feature = band-power change vs the 1.5 s of rest before the cue, in percent (ERD/ERS),
for mu (8-12 Hz) and beta (13-30 Hz) on 7 motor-area electrodes. The rest baseline uses
no labels, so it is allowed on test people.
"""
import glob, json
import numpy as np, mne

W = __file__.rsplit("/repro/", 1)[0]
CH = ["FC3", "FC4", "C3", "Cz", "C4", "CP3", "CP4"]
BANDS = {"mu": (8, 12), "beta": (13, 30)}
SYSTEM = ("You decode motor imagery from EEG. Each line gives the band-power change during 4 s of "
          "imagery versus rest, in percent (negative = desynchronization). Answer with one word: LEFT or RIGHT.")


def trials_of(f):
    raw = mne.io.read_raw_edf(f, preload=True, verbose=0)
    mne.datasets.eegbci.standardize(raw)
    sf = raw.info["sfreq"]
    out = []
    bp = {b: raw.copy().filter(*r, verbose=0).get_data(picks=CH) ** 2 for b, r in BANDS.items()}
    full = raw.copy().filter(8., 30., verbose=0).get_data(picks="eeg")
    for on, desc in zip(raw.annotations.onset, raw.annotations.description):
        if desc not in ("T1", "T2") or on > 100:
            continue
        a, b = int(on * sf), int((on + 4) * sf)
        r0 = max(0, a - int(1.5 * sf))
        feat = {}
        for band, p in bp.items():
            feat[band] = 100 * (p[:, a:b].mean(1) - p[:, r0:a].mean(1)) / (p[:, r0:a].mean(1) + 1e-20)
        out.append(dict(label="LEFT" if desc == "T1" else "RIGHT", feat=feat, epoch=full[:, a:b].astype(np.float32)))
    return out


def to_text(feat):
    lines = [f"{c}: mu {feat['mu'][i]:+.0f}%, beta {feat['beta'][i]:+.0f}%" for i, c in enumerate(CH)]
    lat = feat["mu"][CH.index("C3")] - feat["mu"][CH.index("C4")]
    lines.append(f"mu laterality C3-C4: {lat:+.0f} points")
    return "\n".join(lines)


def main():
    rows = []
    for f in sorted(glob.glob(f"{W}/data/eeg/*.edf")):
        name = f.rsplit("/", 1)[1][:-4]
        person = name[:4]
        split = "test" if person >= "S101" else "train"
        for k, t in enumerate(trials_of(f)):
            rows.append(dict(id=f"{name}_{k}", person=person, split=split, label=t["label"],
                             text=to_text(t["feat"]), feat=np.concatenate([t["feat"]["mu"], t["feat"]["beta"]]),
                             epoch=t["epoch"]))
    for split in ("train", "test"):
        with open(f"{W}/repro/{split}.jsonl", "w") as fh:
            for r in rows:
                if r["split"] == split:
                    fh.write(json.dumps({"messages": [
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": r["text"]},
                        {"role": "assistant", "content": r["label"]}]}) + "\n")
    np.savez_compressed(f"{W}/repro/arrays.npz",
                        ids=[r["id"] for r in rows], split=[r["split"] for r in rows],
                        person=[r["person"] for r in rows], y=[r["label"] == "RIGHT" for r in rows],
                        feat=np.stack([r["feat"] for r in rows]), epoch=np.stack([r["epoch"] for r in rows]))
    for s in ("train", "test"):
        n = [r for r in rows if r["split"] == s]
        print(s, len(n), "trials", len({r['person'] for r in n}), "people", f"{np.mean([r['label']=='RIGHT' for r in n]):.2f} right")
    print("example:\n" + rows[0]["text"], "->", rows[0]["label"])


if __name__ == "__main__":
    main()
