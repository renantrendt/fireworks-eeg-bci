"""Render the exoskeleton demo video: real held-out EEG trials replayed in recording order,
each decoded command drives one arm of the exoskeleton.
"""
import sys
import numpy as np, mujoco, imageio
from PIL import Image, ImageDraw
from panels import *
from exo_scene import ExoScene, action_pose, REST

FPS = 30
RUN = int(sys.argv[1]) if len(sys.argv) > 1 else 0      # which held-out run to replay
NTRIALS = int(sys.argv[2]) if len(sys.argv) > 2 else 7
OUT = sys.argv[3] if len(sys.argv) > 3 else f"{W}/out/exo_demo.mp4"
PRE, IMAG, ACT = 1.0, 4.0, 4.8                           # seconds: rest before cue, imagery, device action
SPEED = 7.0 / ACT                                         # action_pose runs 7 s; play it faster

data = Data()
trials = [(i, t) for i, t in enumerate(data.meta) if t["run"] == RUN][:NTRIALS]
acc_all = np.mean([t["truth"] == t["guess"] for t in data.meta])

sc = ExoScene()
r = mujoco.Renderer(sc.m, 926, 1100)
cam = mujoco.MjvCamera()
cam.lookat[:] = [0.3, 0, 0.95]; cam.elevation = -16; cam.distance = 1.75

seg = PRE + IMAG + ACT
total = seg * len(trials) + 2.0
w = imageio.get_writer(OUT, fps=FPS, codec="libx264", quality=None, macro_block_size=1, ffmpeg_params=["-crf", "18"])
rows = []
for f in range(int(total * FPS)):
    t = f / FPS
    k = min(int(t // seg), len(trials) - 1)
    tl = t - k * seg
    ti, tr = trials[k]
    side_true = "LEFT" if tr["truth"] == "L" else "RIGHT"
    t_eeg = tr["onset"] - PRE + tl
    decoded = tl >= PRE + IMAG
    if decoded and (not rows or rows[-1]["n"] != k + 1):
        rows.append(dict(n=k + 1, run=data.names[RUN][1:], **{x: tr[x] for x in ("truth", "guess", "conf")}))
    # device
    for s in "LR":
        sc.set_pose(s, REST); sc.held[s] = False; sc.highlight(s, False)
    if decoded and t < seg * len(trials):
        ta = (tl - PRE - IMAG) * SPEED
        p, h = action_pose(ta)
        g = tr["guess"]
        sc.set_pose(g, p); sc.held[g] = h
        sc.highlight(g, ta < 6.6, ok=tr["guess"] == tr["truth"])
    sc.apply()
    cam.azimuth = 160 + 25 * np.sin(2 * np.pi * t / 40)
    r.update_scene(sc.d, cam)
    sim = Image.fromarray(r.render())
    # frame
    img = Image.new("RGB", (1920, 1080), BG)
    d = ImageDraw.Draw(img)
    header(d, "Thought → Exoskeleton", "Motor imagery BCI · PhysioNet EEGMMIDB (Schalk et al.) · person never seen in training · "
                                        "imagined LEFT fist drives the left arm, RIGHT fist the right arm")
    eeg_panel(d, data, RUN, t_eeg, (20, 104, 780, 470), win=(tr["onset"], tr["onset"] + IMAG),
              cue=(tr["onset"], side_true), label=f"trial {k + 1} of {len(trials)}")
    topo_panel(img, d, data, ti, (20, 480, 780, 770), decoded)
    log_panel(d, rows, (20, 780, 780, 1030), acc_all, len(data.meta))
    img.paste(sim, (800, 104))
    d.rounded_rectangle([800, 104, 1900, 1030], radius=10, outline=LINE, width=2)
    if decoded:
        ok = tr["guess"] == tr["truth"]
        g = "LEFT" if tr["guess"] == "L" else "RIGHT"
        d.rounded_rectangle([824, 124, 1300, 214], radius=10, fill=(11, 19, 27))
        d.text((844, 132), f"decoded: {g} hand", font=F["big"], fill=GREEN if ok else RED)
        d.text((846, 186), f"cue was {side_true} · {'correct' if ok else 'wrong arm moved'} · conf {tr['conf']:.2f}",
               font=F["s"], fill=DIM)
        status_bar(d, f"Exoskeleton: {g.lower()} hand closes on the cup and lifts it, elbow motor assists flexion.")
    elif tl >= PRE:
        d.rounded_rectangle([824, 124, 1300, 190], radius=10, fill=(11, 19, 27))
        d.text((844, 136), f"imagining… {tl - PRE:.1f} / 4.0 s", font=F["h1"], fill=BLUE)
        status_bar(d, f"Cue: imagine squeezing the {side_true.lower()} fist. Collecting 4 s of EEG.")
    else:
        status_bar(d, "Rest. Waiting for the next cue.", DIM)
    w.append_data(np.asarray(img))
w.close()
print("wrote", OUT, "trials", len(trials), "correct", sum(r["truth"] == r["guess"] for r in rows))
