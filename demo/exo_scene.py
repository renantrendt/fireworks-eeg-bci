"""Exoskeleton scene: a seated person with an EEG cap and powered elbow + hand exoskeleton on both arms.

A decoded LEFT or RIGHT trial drives that side's exoskeleton: the hand closes around the cup,
the elbow lifts it, holds, and puts it back. Kinematic replay (joint angles set per frame).
"""
import numpy as np, mujoco
from person import person, chair, ASSETS, VISUAL

XML = f"""<mujoco model="bci_exo">
  <compiler angle="radian"/>
  {VISUAL}
  <asset>{ASSETS}
    <material name="cup" rgba="0.9 0.92 0.95 0.55" specular="1" shininess="1" reflectance="0.2"/>
    <material name="water" rgba="0.35 0.6 0.85 0.7"/>
  </asset>
  <worldbody>
    <light pos="1.5 -1.5 3" dir="-0.4 0.4 -1" diffuse="0.7 0.7 0.68" castshadow="true"/>
    <light pos="-0.5 1.8 2.6" dir="0.3 -0.5 -1" diffuse="0.3 0.32 0.35" castshadow="false"/>
    <geom type="plane" size="6 6 0.1" material="floor"/>
    <geom type="box" size="0.02 4 1.6" pos="-1.6 0 1.6" material="wall"/>
    <geom type="box" size="4 0.02 1.6" pos="0 2.2 1.6" material="wall2"/>
    {chair("0 0 0")}
    <body pos="0.62 0 0">
      <geom type="box" size="0.32 0.6 0.025" pos="0 0 0.675" material="wood"/>
      <geom type="cylinder" fromto="0.26 0.52 0.65 0.26 0.52 0" size="0.025" material="steel"/>
      <geom type="cylinder" fromto="0.26 -0.52 0.65 0.26 -0.52 0" size="0.025" material="steel"/>
      <geom type="cylinder" fromto="-0.26 0.52 0.65 -0.26 0.52 0" size="0.025" material="steel"/>
      <geom type="cylinder" fromto="-0.26 -0.52 0.65 -0.26 -0.52 0" size="0.025" material="steel"/>
      <geom type="box" size="0.06 0.09 0.025" pos="0.2 0.38 0.725" material="white"/>
      <geom type="box" size="0.002 0.07 0.012" pos="0.14 0.38 0.73" rgba="0.15 0.6 0.9 1"/>
    </body>
    {person("0 0 0", exo=True)}
    <body name="cupL" mocap="true" pos="0.42 0.205 0.745">
      <geom type="cylinder" size="0.036 0.05" material="cup" contype="0" conaffinity="0"/>
      <geom type="cylinder" size="0.032 0.03" pos="0 0 -0.015" material="water" contype="0" conaffinity="0"/>
    </body>
    <body name="cupR" mocap="true" pos="0.42 -0.205 0.745">
      <geom type="cylinder" size="0.036 0.05" material="cup" contype="0" conaffinity="0"/>
      <geom type="cylinder" size="0.032 0.03" pos="0 0 -0.015" material="water" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>"""

REST = dict(ab=0.0, sh=-0.3, el=-0.28, wr=0.05, f=0.25, th=0.15)
GRIP = dict(ab=0.0, sh=-0.3, el=-0.28, wr=0.05, f=1.25, th=0.9)
LIFT = dict(ab=-0.75, sh=-0.9, el=0.95, wr=-0.1, f=1.25, th=0.9)


class ExoScene:
    def __init__(self):
        self.m = mujoco.MjModel.from_xml_string(XML)
        self.d = mujoco.MjData(self.m)
        j = lambda n: self.m.joint(n).qposadr[0]
        self.q = {s: dict(ro=j(f"{s}_ro"), ab=j(f"{s}_ab"), sh=j(f"{s}_sh"), el=j(f"{s}_el"), wr=j(f"{s}_wr"), th=j(f"{s}_th"),
                          f=[j(f"{s}_f{i}{k}") for i in range(4) for k in "ab"]) for s in "LR"}
        self.led = {s: self.m.geom(f"{s}_led").id for s in "LR"}
        self.el = {n: self.m.geom(f"el_{n}").id for n in ("C3", "C4")}
        self.cup = {s: self.m.body(f"cup{s}").mocapid[0] for s in "LR"}
        self.hand = {s: self.m.body(f"{s}_hand").id for s in "LR"}
        self.pose = {s: dict(GRIP) for s in "LR"}
        self.held = {s: True for s in "LR"}
        self.cup0 = {s: np.zeros(3) for s in "LR"}
        self.apply()
        self.cup0 = {s: self.d.mocap_pos[self.cup[s]].copy() * [1, 1, 0] + [0, 0, 0.752] for s in "LR"}
        self.pose = {s: dict(REST) for s in "LR"}
        self.held = {s: False for s in "LR"}
        self.apply()

    def set_pose(self, side, p):
        self.pose[side] = dict(p)

    def apply(self):
        for s in "LR":
            p, q = self.pose[s], self.q[s]
            self.d.qpos[q["sh"]], self.d.qpos[q["el"]], self.d.qpos[q["wr"]] = p["sh"], p["el"], p["wr"]
            self.d.qpos[q["th"]] = p["th"]
            self.d.qpos[q["ab"]] = p.get("ab", 0) * (1 if s == "R" else -1)
            self.d.qpos[q["ro"]] = 1.57 * (1 if s == "R" else -1)
            for a in q["f"]:
                self.d.qpos[a] = p["f"]
        mujoco.mj_forward(self.m, self.d)
        for s in "LR":
            if self.held[s]:
                R = self.d.xmat[self.hand[s]].reshape(3, 3)
                self.d.mocap_pos[self.cup[s]] = self.d.xpos[self.hand[s]] + R @ np.array([0.06, 0, -0.058])
            else:
                self.d.mocap_pos[self.cup[s]] = self.cup0[s]
        mujoco.mj_forward(self.m, self.d)

    def highlight(self, side, on, ok=True):
        """Motor LED and the contralateral electrode (C4 for left hand, C3 for right) light up."""
        col = np.array([0.3, 0.95, 0.5, 1]) if ok else np.array([1, 0.45, 0.2, 1])
        self.m.geom_rgba[self.led[side]] = col if on else [0.2, 0.25, 0.3, 1]
        e = self.el["C4" if side == "L" else "C3"]
        self.m.geom_rgba[e] = [0.35, 0.7, 1, 1] if on else [0.75, 0.78, 0.8, 1]


def lerp(a, b, t):
    t = 0.5 - 0.5 * np.cos(np.pi * np.clip(t, 0, 1))
    return {k: a[k] + (b[k] - a[k]) * t for k in a}


def action_pose(t):
    """Pose and whether the cup is held, t = seconds since the command started (7 s action)."""
    if t < 0.8: return lerp(REST, GRIP, t / 0.8), t > 0.7
    if t < 2.6: return lerp(GRIP, LIFT, (t - 0.8) / 1.8), True
    if t < 4.0: return LIFT, True
    if t < 5.8: return lerp(LIFT, GRIP, (t - 4.0) / 1.8), True
    if t < 6.6: return lerp(GRIP, REST, (t - 5.8) / 0.8), t < 5.9
    return REST, False
