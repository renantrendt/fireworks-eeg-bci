"""Kitchen scene: a Franka Panda arm on a mobile base fetches a water bottle from the fridge
for a seated person who wears an EEG cap.

The person picks the goal with two decoded commands (LEFT = next item, RIGHT = confirm).
The robot then runs the fetch on its own (shared autonomy). Kinematic replay: joint angles
come from damped least-squares IK, then mj_forward.
"""
import os
import numpy as np, mujoco
from person import person, chair, ASSETS, VISUAL

W = __file__.rsplit("/demo/", 1)[0]
PANDA = f"{W}/assets/menagerie/franka_emika_panda/panda.xml"

FRIDGE_X, FRIDGE_Y = 2.75, 1.25   # fridge centre, front face at x = FRIDGE_X - 0.35
TABLE = (0.62, 0.0)

XML = f"""<mujoco model="bci_kitchen">
  <compiler angle="radian"/>
  {VISUAL}
  <asset>{ASSETS}
    <texture name="tile" type="2d" builtin="checker" rgb1="0.9 0.9 0.88" rgb2="0.86 0.86 0.84" width="256" height="256" mark="edge" markrgb="0.8 0.8 0.78"/>
    <material name="tile" texture="tile" texrepeat="10 4"/>
    <material name="counter" rgba="0.2 0.21 0.23 1" specular="0.6" shininess="0.7" reflectance="0.15"/>
    <material name="cabinet" rgba="0.88 0.86 0.82 1"/>
    <material name="fridge" rgba="0.78 0.8 0.82 1" specular="1" shininess="0.95" reflectance="0.35"/>
    <material name="fridge_in" rgba="0.95 0.96 0.97 1"/>
    <material name="glassshelf" rgba="0.8 0.9 0.95 0.35"/>
    <material name="bottle" rgba="0.55 0.78 0.95 0.75" specular="1" shininess="1" reflectance="0.3"/>
    <material name="cap_blue" rgba="0.1 0.35 0.8 1"/>
    <material name="base" rgba="0.92 0.92 0.9 1" specular="0.4"/>
    <material name="rubber" rgba="0.08 0.08 0.09 1"/>
  </asset>
  <worldbody>
    <light pos="1.2 0 3" dir="0 0 -1" diffuse="0.6 0.6 0.58" castshadow="true"/>
    <light pos="2.2 1.0 2.8" dir="0 0 -1" diffuse="0.35 0.35 0.35" castshadow="false"/>
    <light pos="3 -2 2.6" dir="-0.5 0.6 -0.6" diffuse="0.35 0.35 0.35" castshadow="false"/>
    <geom type="plane" size="8 8 0.1" material="floor"/>
    <geom type="box" size="0.02 4 1.5" pos="3.25 0 1.5" material="wall"/>
    <geom type="box" size="4 0.02 1.5" pos="0 2.0 1.5" material="wall2"/>
    <!-- counter run along the back wall -->
    <geom type="box" size="0.8 0.3 0.43" pos="0.65 1.68 0.43" material="cabinet"/>
    <geom type="box" size="0.82 0.32 0.02" pos="0.65 1.68 0.88" material="counter"/>
    <geom type="box" size="0.8 0.01 0.25" pos="0.65 1.98 1.15" material="tile"/>
    <geom type="box" size="0.8 0.18 0.32" pos="0.65 1.8 1.75" material="cabinet"/>
    <geom type="box" size="0.22 0.15 0.005" pos="0.5 1.68 0.9" material="steel"/>
    <geom type="capsule" fromto="0.5 1.9 0.9 0.5 1.9 1.12" size="0.012" material="steel"/>
    <geom type="capsule" fromto="0.5 1.9 1.12 0.5 1.78 1.12" size="0.012" material="steel"/>
    <geom type="cylinder" size="0.1 0.06" pos="1.15 1.68 0.96" material="white"/>
    <geom type="box" size="0.002 0.28 0.38" pos="-0.12 1.68 0.43" rgba="0.75 0.73 0.7 1"/>
    <!-- fridge: open front faces -x, door hinged on the +y edge -->
    <body name="fridge" pos="{FRIDGE_X} {FRIDGE_Y} 0">
      <geom type="box" size="0.35 0.38 0.02" pos="0 0 0.05" material="fridge"/>
      <geom type="box" size="0.35 0.38 0.02" pos="0 0 1.83" material="fridge"/>
      <geom type="box" size="0.02 0.38 0.9" pos="0.33 0 0.95" material="fridge"/>
      <geom type="box" size="0.35 0.02 0.9" pos="0 0.36 0.95" material="fridge"/>
      <geom type="box" size="0.35 0.02 0.9" pos="0 -0.36 0.95" material="fridge"/>
      <geom type="box" size="0.31 0.34 0.005" pos="0.02 0 0.95" material="fridge_in"/>
      <geom type="box" size="0.3 0.335 0.004" pos="0.02 0 0.6" material="glassshelf"/>
      <geom type="box" size="0.3 0.335 0.004" pos="0.02 0 1.3" material="glassshelf"/>
      <geom type="box" size="0.3 0.335 0.004" pos="0.02 0 1.6" material="glassshelf"/>
      <geom type="box" size="0.08 0.12 0.06" pos="0.15 0.15 1.02" rgba="0.95 0.85 0.3 1"/>
      <geom type="box" size="0.06 0.06 0.1" pos="0.18 -0.18 0.71" rgba="0.95 0.95 0.92 1"/>
      <geom type="sphere" size="0.045" pos="0.12 -0.1 1.35" rgba="0.85 0.2 0.15 1"/>
      <geom type="sphere" size="0.045" pos="0.2 -0.02 1.35" rgba="0.4 0.75 0.2 1"/>
      <body name="door" pos="-0.36 0.38 0">
        <joint name="door" axis="0 0 1" range="-2 0"/>
        <geom type="box" size="0.025 0.38 0.9" pos="-0.025 -0.38 0.95" material="fridge"/>
        <geom type="box" size="0.02 0.34 0.8" pos="0.02 -0.38 0.95" material="fridge_in"/>
        <geom type="box" size="0.06 0.3 0.01" pos="0.07 -0.38 0.75" material="glassshelf"/>
        <geom type="capsule" fromto="-0.09 -0.7 0.85 -0.09 -0.7 1.25" size="0.014" material="steel"/>
        <geom type="capsule" fromto="-0.05 -0.7 0.87 -0.09 -0.7 0.87" size="0.01" material="steel"/>
        <geom type="capsule" fromto="-0.05 -0.7 1.23 -0.09 -0.7 1.23" size="0.01" material="steel"/>
        <site name="handle" pos="-0.09 -0.7 1.05"/>
      </body>
    </body>
    <body name="bottle" mocap="true" pos="{FRIDGE_X - 0.12} {FRIDGE_Y} 0.97">
      <geom type="cylinder" size="0.033 0.095" pos="0 0 0.1" material="bottle"/>
      <geom type="cylinder" size="0.018 0.02" pos="0 0 0.21" material="bottle"/>
      <geom type="cylinder" size="0.019 0.012" pos="0 0 0.24" material="cap_blue"/>
      <geom type="cylinder" size="0.034 0.03" pos="0 0 0.09" rgba="0.15 0.45 0.85 1"/>
    </body>
    <!-- dining table and the person -->
    {chair("0 0 0")}
    <body pos="{TABLE[0]} {TABLE[1]} 0">
      <geom type="box" size="0.32 0.6 0.025" pos="0 0 0.675" material="wood"/>
      <geom type="cylinder" fromto="0.26 0.52 0.65 0.26 0.52 0" size="0.025" material="steel"/>
      <geom type="cylinder" fromto="0.26 -0.52 0.65 0.26 -0.52 0" size="0.025" material="steel"/>
      <geom type="cylinder" fromto="-0.26 0.52 0.65 -0.26 0.52 0" size="0.025" material="steel"/>
      <geom type="cylinder" fromto="-0.26 -0.52 0.65 -0.26 -0.52 0" size="0.025" material="steel"/>
      <geom type="box" size="0.06 0.09 0.025" pos="-0.05 0.38 0.725" material="white"/>
      <geom type="box" size="0.002 0.07 0.012" pos="-0.11 0.38 0.73" rgba="0.15 0.6 0.9 1"/>
    </body>
    {person("0 0 0", exo=False)}
    <!-- mobile base; the Panda is attached at site "mount" -->
    <body name="base" mocap="true" pos="1.6 -0.9 0">
      <geom type="box" size="0.3 0.25 0.16" pos="0 0 0.2" material="base"/>
      <geom type="box" size="0.31 0.26 0.02" pos="0 0 0.37" rgba="0.2 0.22 0.25 1"/>
      <geom type="cylinder" size="0.07 0.025" pos="0.2 0.27 0.07" euler="1.5708 0 0" material="rubber"/>
      <geom type="cylinder" size="0.07 0.025" pos="-0.2 0.27 0.07" euler="1.5708 0 0" material="rubber"/>
      <geom type="cylinder" size="0.07 0.025" pos="0.2 -0.27 0.07" euler="1.5708 0 0" material="rubber"/>
      <geom type="cylinder" size="0.07 0.025" pos="-0.2 -0.27 0.07" euler="1.5708 0 0" material="rubber"/>
      <geom type="box" size="0.02 0.18 0.03" pos="0.3 0 0.3" rgba="0.1 0.12 0.14 1"/>
      <geom name="base_led" type="box" size="0.003 0.15 0.006" pos="0.32 0 0.3" rgba="0.3 0.9 0.5 1"/>
      <geom type="cylinder" size="0.04 0.08" pos="-0.18 0 0.47" rgba="0.15 0.16 0.18 1"/>
      <geom type="box" size="0.03 0.05 0.03" pos="-0.18 0 0.57" rgba="0.1 0.1 0.1 1"/>
      <site name="mount" pos="0.08 0 0.39"/>
    </body>
  </worldbody>
</mujoco>"""

ARM = [f"p_joint{i}" for i in range(1, 8)]
HOME = np.array([0, -0.5, 0, -2.2, 0, 1.7, 0.785])


def build():
    spec = mujoco.MjSpec.from_string(XML)
    panda = mujoco.MjSpec.from_file(PANDA)
    for k in list(panda.keys):
        panda.delete(k)
    spec.attach(panda, prefix="p_", site=spec.site("mount"))
    m = spec.compile()
    return m


class KitchenScene:
    def __init__(self):
        self.m = build()
        self.d = mujoco.MjData(self.m)
        self.qa = np.array([self.m.joint(n).qposadr[0] for n in ARM])
        self.da = np.array([self.m.joint(n).dofadr[0] for n in ARM])
        self.fing = [self.m.joint(f"p_finger_joint{i}").qposadr[0] for i in (1, 2)]
        self.door = self.m.joint("door").qposadr[0]
        self.hand = self.m.body("p_hand").id
        self.base = self.m.body("base").mocapid[0]
        self.bottle = self.m.body("bottle").mocapid[0]
        self.bottle0 = self.d.mocap_pos[self.bottle].copy()
        self.handle = self.m.site("handle").id
        self.led = self.m.geom("base_led").id
        self.el = {n: self.m.geom(f"el_{n}").id for n in ("C3", "C4")}
        self.d.qpos[self.qa] = HOME
        self.grip(0.04)
        self.held = False
        self.set_base(1.6, -0.9, np.pi / 2)

    # --- state setters -------------------------------------------------
    def set_base(self, x, y, yaw):
        self.d.mocap_pos[self.base] = [x, y, 0]
        self.d.mocap_quat[self.base] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        self.yaw = yaw

    def grip(self, w):
        for a in self.fing:
            self.d.qpos[a] = w

    def set_door(self, ang):
        self.d.qpos[self.door] = ang

    def tcp(self):
        R = self.d.xmat[self.hand].reshape(3, 3)
        return self.d.xpos[self.hand] + R[:, 2] * 0.103, R

    def forward(self):
        mujoco.mj_forward(self.m, self.d)
        if self.held:
            p, R = self.tcp()
            self.d.mocap_pos[self.bottle] = p - np.array([0, 0, 0.1])
            mujoco.mj_forward(self.m, self.d)

    def ik(self, target, approach, iters=60):
        """Damped least squares: TCP to `target`, gripper z-axis along `approach`, fingers horizontal."""
        approach = approach / np.linalg.norm(approach)
        side = np.cross([0, 0, 1], approach); side /= np.linalg.norm(side)
        Rt = np.column_stack([side, np.cross(approach, side), approach])
        jp, jr = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv))
        for _ in range(iters):
            mujoco.mj_kinematics(self.m, self.d)
            mujoco.mj_comPos(self.m, self.d)
            p, R = self.tcp()
            ep = target - p
            er = 0.5 * sum(np.cross(R[:, i], Rt[:, i]) for i in range(3))
            if np.linalg.norm(ep) < 1e-4 and np.linalg.norm(er) < 1e-3:
                break
            mujoco.mj_jac(self.m, self.d, jp, jr, p, self.hand)
            J = np.vstack([jp[:, self.da], jr[:, self.da] * 0.5])
            e = np.concatenate([ep, er * 0.5])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(6), e)
            q = self.d.qpos[self.qa] + np.clip(dq, -0.2, 0.2)
            lo, hi = self.m.jnt_range[[self.m.joint(n).id for n in ARM]].T
            self.d.qpos[self.qa] = np.clip(q, lo + 1e-3, hi - 1e-3)
        return np.linalg.norm(ep)

    def highlight(self, side, on):
        e = self.el["C4" if side == "L" else "C3"]
        self.m.geom_rgba[e] = [0.35, 0.7, 1, 1] if on else [0.75, 0.78, 0.8, 1]
