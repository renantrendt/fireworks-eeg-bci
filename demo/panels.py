"""2D overlay panels drawn with PIL: header, EEG traces, scalp map, trial log, accuracy, menu."""
import io, json
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W = __file__.rsplit("/demo/", 1)[0]
BG, PANEL, LINE = (15, 26, 36), (22, 36, 49), (42, 59, 76)
TXT, DIM = (236, 238, 240), (150, 170, 188)
BLUE, ORANGE, GREEN, RED = (90, 169, 230), (242, 140, 56), (123, 211, 137), (240, 110, 90)
CH_COL = [BLUE, TXT, ORANGE]


def font(size, bold=False, mono=False):
    if mono:
        return ImageFont.truetype("/System/Library/Fonts/SFNSMono.ttf", size)
    return ImageFont.truetype("/System/Library/Fonts/HelveticaNeue.ttc", size, index=1 if bold else 0)


F = dict(h1=ImageFont.truetype("/System/Library/Fonts/SFNS.ttf", 30), h2=font(22, True), b=font(20), s=font(17), m=font(18, mono=True),
         ms=font(15, mono=True), big=font(46, True), lab=font(15, True))


class Data:
    def __init__(self):
        z = np.load(f"{W}/data/trials.npz", allow_pickle=True)
        self.disp, self.sf = z["disp"], float(z["sfreq"])
        self.names = list(z["run_names"])
        self.meta = json.loads(str(z["meta"]))
        self.erd = z["erd"]
        self.topo = {}

    def topomap(self, i, size=230):
        if i not in self.topo:
            import matplotlib; matplotlib.use("Agg")
            import matplotlib.pyplot as plt, mne
            info = mne.io.read_info(f"{W}/data/info-info.fif", verbose=0)
            fig, ax = plt.subplots(figsize=(2.4, 2.4), dpi=100)
            fig.patch.set_facecolor(np.array(PANEL) / 255)
            mne.viz.plot_topomap(np.clip(self.erd[i], -60, 60), info, axes=ax, show=False, cmap="RdBu_r",
                                 vlim=(-60, 60), contours=4, sensors=True,
                                 names=None, mask=np.isin(info.ch_names, ["C3", "C4"]),
                                 mask_params=dict(marker="o", markerfacecolor="w", markeredgecolor="k", markersize=7))
            buf = io.BytesIO(); fig.savefig(buf, format="png", facecolor=fig.get_facecolor()); plt.close(fig)
            self.topo[i] = Image.open(buf).convert("RGB").resize((size, size), Image.LANCZOS)
        return self.topo[i]


DECODER_LABEL = "DECODER: CSP+LDA (REAL PREDICTIONS)"


def header(d, title, sub):
    d.rectangle([0, 0, 1920, 64], fill=(11, 19, 27))
    d.text((28, 15), title, font=F["h1"], fill=TXT)
    x = 1892
    for txt, col in [(DECODER_LABEL, ORANGE), (f"DEVICE: SIMULATED (MuJoCo {__import__('mujoco').__version__})", BLUE),
                     ("EEG: REAL, HELD-OUT PERSON", GREEN)]:
        w = d.textlength(txt, font=F["lab"]) + 24
        x -= w
        d.rounded_rectangle([x, 18, x + w - 8, 46], radius=6, outline=col, width=2)
        d.text((x + 8, 23), txt, font=F["lab"], fill=col)
        x -= 8
    d.text((28, 70), sub, font=F["s"], fill=DIM)


def eeg_panel(d, data, run, t_eeg, box, win=(None, None), cue=None, label=""):
    """Scroll C3 / Cz / C4 for the 6 s ending at t_eeg. win = (start, end) of the decoding window in run seconds."""
    x0, y0, x1, y1 = box
    d.rounded_rectangle(box, radius=10, fill=PANEL)
    d.text((x0 + 16, y0 + 8), f"EEG  ·  {data.names[run]}  ·  1–40 Hz  ·  {label}", font=F["h2"], fill=TXT)
    span, sf = 6.0, data.sf
    pw, top = x1 - x0 - 110, y0 + 66
    ph = (y1 - top - 40) / 3
    a = t_eeg - span
    tx = lambda t: x0 + 70 + (t - a) / span * pw
    if win[0] is not None:
        wa, wb = max(win[0], a), min(win[1], t_eeg)
        if wb > wa:
            d.rectangle([tx(wa), top, tx(wb), y1 - 34], fill=(34, 56, 76))
            d.text((tx(wa) + 6, y1 - 58), "decoding window (4 s)", font=F["s"], fill=BLUE)
    if cue is not None and a <= cue[0] <= t_eeg:
        d.line([tx(cue[0]), top, tx(cue[0]), y1 - 34], fill=ORANGE, width=2)
        d.text((tx(cue[0]) + 6, top - 22), f"cue: imagine {cue[1]} fist", font=F["s"], fill=ORANGE)
    for g in range(7):
        gx = tx(np.floor(a) + g)
        if x0 + 70 <= gx <= x0 + 70 + pw:
            d.line([gx, top, gx, y1 - 34], fill=LINE, width=1)
            d.text((gx - 10, y1 - 28), f"{np.floor(a) + g:.0f}s", font=F["ms"], fill=DIM)
    i0, i1 = int(max(a, 0) * sf), int(t_eeg * sf)
    for c, name in enumerate(["C3", "Cz", "C4"]):
        mid = top + ph * (c + 0.5)
        d.text((x0 + 16, mid - 10), name, font=F["m"], fill=CH_COL[c])
        seg = data.disp[run, c, i0:i1]
        if len(seg) > 2:
            ts = np.arange(i0, i1) / sf
            xs, ys = tx(ts), mid - np.clip(seg, -100, 100) / 100 * (ph * 0.48)
            d.line(list(zip(xs.tolist(), ys.tolist())), fill=CH_COL[c], width=1)
    d.line([x1 - 30, top + 10, x1 - 30, top + 10 + ph * 0.24], fill=DIM, width=2)
    d.text((x1 - 26, top + 6), "50µV", font=F["ms"], fill=DIM)


def topo_panel(img, d, data, ti, box, shown):
    x0, y0, x1, y1 = box
    d.rounded_rectangle(box, radius=10, fill=PANEL)
    d.text((x0 + 16, y0 + 10), "Mu power (8–12 Hz) vs rest", font=F["h2"], fill=TXT)
    if shown and ti is not None:
        img.paste(data.topomap(ti), (x0 + 20, y0 + 44))
        d.text((x0 + 262, y0 + 60), "blue = power drop", font=F["s"], fill=BLUE)
        d.text((x0 + 262, y0 + 84), "red = power rise", font=F["s"], fill=RED)
        d.text((x0 + 262, y0 + 124), "Imagined LEFT hand:", font=F["s"], fill=DIM)
        d.text((x0 + 262, y0 + 146), "drop over C4 (right)", font=F["s"], fill=DIM)
        d.text((x0 + 262, y0 + 176), "Imagined RIGHT hand:", font=F["s"], fill=DIM)
        d.text((x0 + 262, y0 + 198), "drop over C3 (left)", font=F["s"], fill=DIM)
        d.text((x0 + 262, y0 + 236), "real data, this trial", font=F["s"], fill=GREEN)
    else:
        d.text((x0 + 20, y0 + 130), "computed after the 4 s window", font=F["s"], fill=DIM)


def log_panel(d, rows, box, acc_all, n_all):
    x0, y0, x1, y1 = box
    d.rounded_rectangle(box, radius=10, fill=PANEL)
    d.text((x0 + 16, y0 + 10), "Trial log", font=F["h2"], fill=TXT)
    hdr = ["#", "run", "cue", "decoded", "conf", "result"]
    cx = [x0 + 16, x0 + 60, x0 + 170, x0 + 250, x0 + 360, x0 + 440]
    for h, x in zip(hdr, cx):
        d.text((x, y0 + 46), h, font=F["lab"], fill=DIM)
    for k, r in enumerate(rows[-6:]):
        y = y0 + 74 + k * 28
        ok = r["truth"] == r["guess"]
        vals = [str(r["n"]), r["run"], "LEFT" if r["truth"] == "L" else "RIGHT",
                "LEFT" if r["guess"] == "L" else "RIGHT", f'{r["conf"]:.2f}', "correct" if ok else "wrong"]
        for j, (v, x) in enumerate(zip(vals, cx)):
            d.text((x, y), v, font=F["m"], fill=(GREEN if ok else RED) if j == 5 else TXT)
    n = len(rows); c = sum(r["truth"] == r["guess"] for r in rows)
    yb = y1 - 72
    d.text((x0 + 16, yb), f"this session {c}/{n}", font=F["b"], fill=TXT)
    d.text((x0 + 16, yb + 30), f"all {n_all} held-out trials: {acc_all:.1%}", font=F["s"], fill=ORANGE)
    bx0, bx1 = x0 + 330, x1 - 20
    d.rectangle([bx0, yb + 4, bx1, yb + 24], fill=LINE)
    if n:
        d.rectangle([bx0, yb + 4, bx0 + (bx1 - bx0) * c / n, yb + 24], fill=GREEN)
    xm = bx0 + (bx1 - bx0) * 0.5
    d.line([xm, yb - 2, xm, yb + 30], fill=TXT, width=2)
    d.text((xm - 30, yb + 32), "chance", font=F["ms"], fill=DIM)


def status_bar(d, text, col=TXT):
    d.rectangle([0, 1040, 1920, 1080], fill=(11, 19, 27))
    d.text((28, 1049), text, font=F["b"], fill=col)
