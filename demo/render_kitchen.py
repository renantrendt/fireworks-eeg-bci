"""Render the kitchen demo: the person picks "Water" from a 3-item menu with decoded imagery
(LEFT = move to next item, RIGHT = select), then the robot fetches the bottle from the fridge.

Each menu command replays the next real held-out trial, in recording order, whose cue matches
the command the person needs. The decoder's guess is applied as-is,
so wrong decodes move the cursor the wrong way.
"""
import sys
import numpy as np, mujoco, imageio
from PIL import Image, ImageDraw
from panels import *
from kitchen_scene import KitchenScene
from fetch_plan import FetchPlan

FPS = 30
PERSON = sys.argv[1] if len(sys.argv) > 1 else "S103"
OUT = f"{W}/out/kitchen_demo.mp4"
ITEMS = ["Apple", "Water", "Medicine"]
PRE, IMAG, SHOW = 1.0, 4.0, 1.6

data = Data()
acc_all = np.mean([t["truth"] == t["guess"] for t in data.meta])
pool = [(i, t) for i, t in enumerate(data.meta) if data.names[t["run"]].startswith(PERSON)]


def plan_menu():
    """LEFT = next, RIGHT = select. After a select, the system asks to confirm:
    RIGHT = confirm, LEFT = cancel. Each step: (trial idx, trial, cursor before, cursor after, selected, state)."""
    cur, used, steps, pend = 0, set(), [], None
    while len(steps) < 14:
        need = ("R" if pend == "Water" else "L") if pend else ("R" if ITEMS[cur] == "Water" else "L")
        i, t = next((i, t) for i, t in pool if t["truth"] == need and i not in used)
        used.add(i)
        before, g = cur, t["guess"]
        if pend:
            if g == "R":
                steps.append((i, t, before, cur, pend, "confirmed"))
                return steps
            steps.append((i, t, before, cur, None, "cancelled"))
            pend = None
        elif g == "L":
            cur = (cur + 1) % len(ITEMS)
            steps.append((i, t, before, cur, None, "next"))
        else:
            pend = ITEMS[cur]
            steps.append((i, t, before, cur, pend, "confirm?"))
    return steps


OUTCOMES = "All 4 test people, same decoder: S103 water in 3 commands · S102 water in 9 · S104 confirmed Apple by mistake · S101 no selection in 14"
steps = plan_menu()
assert steps[-1][5] == "confirmed" and steps[-1][4] == "Water", f"{PERSON}: {steps[-1]}"

k = KitchenScene()
plan = FetchPlan(k)
r = mujoco.Renderer(k.m, 926, 1100)
cam = mujoco.MjvCamera()
CAMS = {  # time since robot start -> (lookat, azimuth, elevation, distance)
    -99: ([1.3, 0.2, 0.85], 28, -20, 3.7),
    0.0: ([1.3, 0.2, 0.85], 28, -20, 3.7),
    3.0: ([2.05, 1.0, 1.0], 22, -14, 2.5),
    16.5: ([2.05, 1.0, 1.0], 22, -14, 2.5),
    19.5: ([1.0, -0.2, 0.85], 55, -22, 2.6),
    26.0: ([1.0, -0.2, 0.85], 75, -24, 2.4),
}


def set_cam(tr):
    ks = sorted(CAMS)
    j = max(i for i, x in enumerate(ks) if x <= tr)
    a, b = CAMS[ks[j]], CAMS[ks[min(j + 1, len(ks) - 1)]]
    s = 0 if j == len(ks) - 1 else np.clip((tr - ks[j]) / (ks[j + 1] - ks[j]), 0, 1)
    s = 0.5 - 0.5 * np.cos(np.pi * s)
    cam.lookat[:] = np.array(a[0]) + (np.array(b[0]) - a[0]) * s
    cam.azimuth, cam.elevation, cam.distance = [a[i] + (b[i] - a[i]) * s for i in (1, 2, 3)]


def menu_panel(d, box, cur, selected, phase_text, progress, pending=None):
    x0, y0, x1, y1 = box
    d.rounded_rectangle(box, radius=10, fill=PANEL)
    d.text((x0 + 16, y0 + 10), "BCI menu", font=F["h2"], fill=TXT)
    d.text((x0 + 150, y0 + 14), "imagine LEFT = next item · imagine RIGHT = select", font=F["s"], fill=DIM)
    for j, it in enumerate(ITEMS):
        bx = x0 + 20 + j * 248
        on = j == cur
        fill = (34, 56, 76) if on else BG
        col = GREEN if (selected == it) else (BLUE if on else LINE)
        d.rounded_rectangle([bx, y0 + 52, bx + 228, y0 + 142], radius=10, fill=fill, outline=col, width=3 if on else 1)
        d.text((bx + 20, y0 + 78), it, font=F["h1"], fill=TXT if on else DIM)
    if selected:
        d.text((x0 + 20, y0 + 160), f"selected: {selected}", font=F["h2"], fill=GREEN)
        d.text((x0 + 20, y0 + 196), f"robot: {phase_text}", font=F["b"], fill=TXT)
        d.rectangle([x0 + 20, y0 + 236, x1 - 20, y0 + 254], fill=LINE)
        d.rectangle([x0 + 20, y0 + 236, x0 + 20 + (x1 - x0 - 40) * progress, y0 + 254], fill=BLUE)
        d.text((x0 + 20, y0 + 260), "autonomous fetch: the brain picks the goal, the robot plans the motion",
               font=F["s"], fill=DIM)
    elif pending:
        d.text((x0 + 20, y0 + 166), f"confirm {pending}?  RIGHT = yes · LEFT = cancel", font=F["h2"], fill=ORANGE)
    else:
        d.text((x0 + 20, y0 + 166), "goal: get a bottle of water from the fridge", font=F["b"], fill=DIM)


seg = PRE + IMAG + SHOW
t_menu = seg * len(steps)
total = t_menu + plan.end + 2.5
w = imageio.get_writer(OUT, fps=FPS, codec="libx264", quality=None, macro_block_size=1, ffmpeg_params=["-crf", "18"])
rows = []
for f in range(int(total * FPS)):
    t = f / FPS
    img = Image.new("RGB", (1920, 1080), BG)
    d = ImageDraw.Draw(img)
    header(d, "Thought → Assistant robot", "Motor imagery BCI · PhysioNet EEGMMIDB (Schalk et al.) · person never seen in training · "
                                           "Franka Panda (MuJoCo Menagerie) · " + OUTCOMES)
    if t < t_menu:
        j = int(t // seg); tl = t - j * seg
        ti, tr, before, after, sel, state = steps[j]
        decoded = tl >= PRE + IMAG
        if decoded and (not rows or rows[-1]["n"] != j + 1):
            rows.append(dict(n=j + 1, run=data.names[tr["run"]][1:], **{x: tr[x] for x in ("truth", "guess", "conf")}))
        need = "RIGHT" if tr["truth"] == "R" else "LEFT"
        side_true = "LEFT" if tr["truth"] == "L" else "RIGHT"
        eeg_panel(d, data, tr["run"], tr["onset"] - PRE + tl, (20, 104, 780, 470), win=(tr["onset"], tr["onset"] + IMAG),
                  cue=(tr["onset"], side_true), label=f"command {j + 1}")
        cur = after if decoded else before
        menu_panel(d, (20, 480, 780, 770), cur, sel if decoded and state == "confirmed" else None, "", 0, pending=sel if decoded and state == "confirm?" else None)
        plan.step(0)
        k.highlight(tr["guess"], decoded)
        set_cam(-1)
        if decoded:
            ok = tr["guess"] == tr["truth"]
            status_bar(d, f"Needed {need}. Decoded {'LEFT' if tr['guess'] == 'L' else 'RIGHT'}: "
                          f"{'correct' if ok else 'wrong'} → {state}.", GREEN if ok else RED)
        elif tl >= PRE:
            status_bar(d, f"Command {j + 1}: the person needs {need}. Collecting 4 s of EEG… {tl - PRE:.1f} s")
        else:
            status_bar(d, "Rest.", DIM)
    else:
        tr_ = t - t_menu
        k.highlight("L", False); k.highlight("R", False)
        plan.step(min(tr_, plan.end))
        set_cam(tr_)
        last = steps[-1][1]
        eeg_panel(d, data, last["run"], min(last["onset"] + IMAG + SHOW + tr_, 105.0), (20, 104, 780, 470),
                  label="monitoring")
        menu_panel(d, (20, 480, 780, 770), 1, "Water", plan.phase, min(tr_ / plan.end, 1))
        status_bar(d, f"Robot: {plan.phase}.")
    log_panel(d, rows, (20, 780, 780, 1030), acc_all, len(data.meta))
    r.update_scene(k.d, cam)
    img.paste(Image.fromarray(r.render()), (800, 104))
    d.rounded_rectangle([800, 104, 1900, 1030], radius=10, outline=LINE, width=2)
    w.append_data(np.asarray(img))
w.close()
print("wrote", OUT, "commands", len(steps), "correct", sum(r["truth"] == r["guess"] for r in rows))
