"""Scripted fetch for the kitchen robot: open fridge, take the water bottle, place it by the person.

`FetchPlan(k).step(t)` sets the scene state for time t (seconds since the robot started).
Call it with increasing t: the IK warm-starts from the previous frame.
"""
import numpy as np

X = np.array([1.0, 0, 0])


def ease(t):
    t = np.clip(t, 0, 1)
    return 0.5 - 0.5 * np.cos(np.pi * t)


class FetchPlan:
    def __init__(self, k):
        self.k = k
        self.start = (1.6, -0.9, np.pi / 2)
        self.at_door = (1.5, 0.75, 0.35)
        self.at_door2 = (1.4, 1.1, 1.1)
        self.at_shelf = (1.9, 1.25, 0.0)
        self.at_table = (1.38, -0.22, np.pi)
        b0 = k.bottle0
        self.bottle_tcp = b0 + [0, 0, 0.1]
        self.place_tcp = np.array([0.74, -0.22, 0.81])
        # (t_end, phase) — phases are handled in step()
        self.T = dict(drive1=3.5, reach_h=5.0, grasp_h=5.5, pull=8.5, rel_h=9.0, home1=10.0,
                      drive2=11.5, pre=13.0, reach_b=14.0, grasp_b=14.6, lift=16.0, carry=17.0,
                      drive3=21.0, place=23.0, release=23.6, home2=25.0)
        self.end = 25.5
        self.qhome = None
        self.phase = ""

    def base(self, a, b, s):
        s = ease(s)
        x = a[0] + (b[0] - a[0]) * s
        y = a[1] + (b[1] - a[1]) * s
        dyaw = (b[2] - a[2] + np.pi) % (2 * np.pi) - np.pi
        self.k.set_base(x, y, a[2] + dyaw * s)

    def handle_pos(self):
        return self.k.d.site_xpos[self.k.handle].copy()

    def move_tcp(self, a, b, s, approach=X):
        self.k.ik(a + (b - a) * ease(s), approach, iters=40)

    def step(self, t):
        k, T = self.k, self.T
        if self.qhome is None:
            self.qhome = k.d.qpos[k.qa].copy()
        k.forward()
        if t < T["drive1"]:
            self.phase = "driving to the fridge"
            self.base(self.start, self.at_door, t / T["drive1"])
            k.d.qpos[k.qa] = self.qhome
        elif t < T["reach_h"]:
            self.phase = "reaching the door handle"
            if not hasattr(self, "p0"):
                k.forward(); self.p0 = k.tcp()[0].copy()
            h = self.handle_pos() + [-0.02, 0, 0]
            self.move_tcp(self.p0, h, (t - T["drive1"]) / (T["reach_h"] - T["drive1"]))
        elif t < T["grasp_h"]:
            self.phase = "grasping the handle"
            k.grip(0.04 - 0.03 * ease((t - T["reach_h"]) / 0.5))
        elif t < T["pull"]:
            self.phase = "opening the fridge"
            s = ease((t - T["grasp_h"]) / (T["pull"] - T["grasp_h"]))
            ang = -1.45 * s
            self.base(self.at_door, self.at_door2, s)
            k.set_door(ang)
            k.forward()
            n = np.array([np.cos(ang), np.sin(ang), 0])  # door face normal (outward is -x at 0)
            self.k.ik(self.handle_pos() + [-0.02 * np.cos(ang), -0.02 * np.sin(ang), 0],
                      np.array([np.cos(ang), np.sin(ang), 0]), iters=40)
        elif t < T["rel_h"]:
            self.phase = "releasing the handle"
            k.grip(0.01 + 0.03 * ease((t - T["pull"]) / 0.5))
            if not hasattr(self, "qr"):
                self.qr = k.d.qpos[k.qa].copy()
        elif t < T["home1"]:
            s = ease((t - T["rel_h"]) / (T["home1"] - T["rel_h"]))
            k.d.qpos[k.qa] = self.qr + (self.qhome - self.qr) * s
        elif t < T["drive2"]:
            self.phase = "moving to the shelf"
            self.base(self.at_door2, self.at_shelf, (t - T["home1"]) / (T["drive2"] - T["home1"]))
        elif t < T["pre"]:
            self.phase = "reaching for the water bottle"
            if not hasattr(self, "p1"):
                k.forward(); self.p1 = k.tcp()[0].copy()
            self.move_tcp(self.p1, self.bottle_tcp - [0.12, 0, 0], (t - T["drive2"]) / (T["pre"] - T["drive2"]))
        elif t < T["reach_b"]:
            self.move_tcp(self.bottle_tcp - [0.12, 0, 0], self.bottle_tcp, (t - T["pre"]) / (T["reach_b"] - T["pre"]))
        elif t < T["grasp_b"]:
            self.phase = "grasping the bottle"
            k.grip(0.04 - 0.008 * ease((t - T["reach_b"]) / 0.6))
            k.held = t > T["reach_b"] + 0.4
        elif t < T["lift"]:
            self.phase = "taking the bottle out"
            self.move_tcp(self.bottle_tcp, self.bottle_tcp + [-0.42, 0, 0.06], (t - T["grasp_b"]) / (T["lift"] - T["grasp_b"]))
        elif t < T["carry"]:
            if not hasattr(self, "qc"):
                self.qc = k.d.qpos[k.qa].copy()
                self.qcarry = self.qhome.copy()
            s = ease((t - T["lift"]) / (T["carry"] - T["lift"]))
            k.d.qpos[k.qa] = self.qc + (self.qcarry - self.qc) * s
        elif t < T["drive3"]:
            self.phase = "bringing the bottle to the table"
            self.base(self.at_shelf, self.at_table, (t - T["carry"]) / (T["drive3"] - T["carry"]))
        elif t < T["place"]:
            self.phase = "placing the bottle"
            if not hasattr(self, "p2"):
                k.forward(); self.p2 = k.tcp()[0].copy()
            self.move_tcp(self.p2, self.place_tcp, (t - T["drive3"]) / (T["place"] - T["drive3"]), approach=-X)
        elif t < T["release"]:
            k.grip(0.032 + 0.008 * ease((t - T["place"]) / 0.6))
            if k.held:
                k.forward()
                k.held = False
                self.placed = k.d.mocap_pos[k.bottle].copy()
            k.d.mocap_pos[k.bottle] = self.placed
        elif t < T["home2"]:
            self.phase = "done: water delivered"
            if not hasattr(self, "qe"):
                self.qe = k.d.qpos[k.qa].copy()
            s = ease((t - T["release"]) / (T["home2"] - T["release"]))
            k.d.qpos[k.qa] = self.qe + (self.qhome - self.qe) * s
            k.d.mocap_pos[k.bottle] = self.placed
        else:
            self.phase = "done: water delivered"
            k.d.mocap_pos[k.bottle] = self.placed
        k.forward()
