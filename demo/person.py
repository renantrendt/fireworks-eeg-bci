"""MJCF text for a seated person wearing an EEG cap, with an optional arm exoskeleton.

The arm chain is: shoulder (pitch) -> elbow (flexion) -> hand (4 fingers x 2 links + thumb).
Joint names carry the side prefix L_ or R_ so the demo can drive each arm.
"""
import math

SKIN = "0.80 0.62 0.50 1"


def finger(side, i, y, exo):
    plates = ""
    if exo:
        plates = ('<geom type="box" size="0.013 0.008 0.003" pos="0.015 0 0.009" material="carbon" contype="0" conaffinity="0"/>')
    return f"""
      <body name="{side}_f{i}a" pos="0.088 {y} 0">
        <joint name="{side}_f{i}a" axis="0 1 0" range="0 1.6"/>
        <geom type="capsule" fromto="0 0 0 0.032 0 0" size="0.0085" material="skin"/>{plates}
        <body name="{side}_f{i}b" pos="0.034 0 0">
          <joint name="{side}_f{i}b" axis="0 1 0" range="0 1.7"/>
          <geom type="capsule" fromto="0 0 0 0.024 0 0" size="0.0078" material="skin"/>{plates.replace('0.015 0 0.009', '0.011 0 0.008')}
        </body>
      </body>"""


def arm(side, exo):
    s = 1 if side == "L" else -1
    inner = -s  # thumb points toward the body midline
    fingers = "".join(finger(side, i, y * s, exo) for i, y in enumerate([0.027, 0.009, -0.009, -0.027]))
    up_exo = fore_exo = elbow_exo = ""
    if exo:
        up_exo = """
      <geom type="capsule" fromto="0 0 -0.05 0 0 -0.22" size="0.056" material="carbon" contype="0" conaffinity="0"/>
      <geom type="cylinder" fromto="0 0 -0.09 0 0 -0.105" size="0.06" material="strap" contype="0" conaffinity="0"/>
      <geom type="cylinder" fromto="0 0 -0.19 0 0 -0.205" size="0.06" material="strap" contype="0" conaffinity="0"/>
      <geom type="box" size="0.012 0.02 0.09" pos="0 {o} -0.14" material="alu" contype="0" conaffinity="0"/>""".replace("{o}", f"{0.06*s:.3f}")
        elbow_exo = f"""
        <geom name="{side}_motor" type="cylinder" fromto="0 {0.045*s:.3f} 0 0 {0.105*s:.3f} 0" size="0.048" material="motor" contype="0" conaffinity="0"/>
        <geom name="{side}_led" type="cylinder" fromto="0 {0.105*s:.3f} 0 0 {0.109*s:.3f} 0" size="0.02" rgba="0.2 0.25 0.3 1" contype="0" conaffinity="0"/>"""
        fore_exo = f"""
        <geom type="capsule" fromto="0.04 0 0 0.21 0 0" size="0.05" material="carbon" contype="0" conaffinity="0"/>
        <geom type="cylinder" fromto="0.17 0 0 0.185 0 0" size="0.054" material="strap" contype="0" conaffinity="0"/>
        <geom type="box" size="0.08 0.012 0.02" pos="0.1 {0.055*s:.3f} 0" material="alu" contype="0" conaffinity="0"/>"""
    return f"""
    <body name="{side}_shoulder" pos="0 {0.205*s:.3f} 1.02">
      <joint name="{side}_ab" axis="1 0 0" range="-0.8 0.8"/>
      <joint name="{side}_sh" axis="0 1 0" range="-1.4 0.6"/>
      <geom type="sphere" size="0.046" material="shirt"/>
      <geom type="capsule" fromto="0 0 0 0 0 -0.27" size="0.039" material="shirt"/>{up_exo}
      <body name="{side}_elbow" pos="0 0 -0.285">
        <joint name="{side}_el" axis="0 -1 0" range="-0.6 2.4"/>
        <geom type="sphere" size="0.04" material="skin"/>
        <geom type="capsule" fromto="0 0 0 0.25 0 0" size="0.034" material="skin"/>{elbow_exo}{fore_exo}
        <body name="{side}_hand" pos="0.27 0 0">
          <joint name="{side}_ro" axis="1 0 0" range="-1.8 1.8"/>
          <joint name="{side}_wr" axis="0 1 0" range="-0.8 0.8"/>
          <geom type="box" size="0.045 0.042 0.014" pos="0.045 0 0" material="skin"/>
          {'<geom type="box" size="0.04 0.04 0.004" pos="0.045 0 0.018" material="carbon" contype="0" conaffinity="0"/>' if exo else ''}
          <body name="{side}_th" pos="0.03 {0.045*inner:.3f} -0.004" euler="0 0 {math.degrees(0.7)*inner:.1f}">
            <joint name="{side}_th" axis="0 1 0" range="0 1.2"/>
            <geom type="capsule" fromto="0 0 0 0.04 0 0" size="0.009" material="skin"/>
          </body>{fingers}
        </body>
      </body>
    </body>"""


def eeg_cap():
    """Electrodes on the head, 10-20 style. C3 and C4 get names so they can light up."""
    out = []
    pts = [("Cz", 0, 90), ("C3", 90, 50), ("C4", -90, 50), ("Fz", 0, 55), ("Pz", 180, 55),
           ("F3", 50, 40), ("F4", -50, 40), ("P3", 130, 40), ("P4", -130, 40),
           ("T7", 90, 12), ("T8", -90, 12), ("Fp1", 20, 12), ("Fp2", -20, 12), ("O1", 160, 12), ("O2", -160, 12),
           ("FC3", 70, 50), ("FC4", -70, 50), ("CP3", 110, 50), ("CP4", -110, 50)]
    r = 0.104
    for n, az, el in pts:
        a, e = math.radians(az), math.radians(el)
        x, y, z = r * math.cos(e) * math.cos(a), r * math.cos(e) * math.sin(a), r * math.sin(e)
        out.append(f'<geom name="el_{n}" type="sphere" size="0.0105" pos="{x:.4f} {y:.4f} {z:.4f}" rgba="0.75 0.78 0.8 1" contype="0" conaffinity="0"/>')
    return "\n        ".join(out)


def person(pos="0 0 0", exo=True, name="person"):
    return f"""
  <body name="{name}" pos="{pos}">
    <geom type="ellipsoid" size="0.13 0.17 0.08" pos="0 0 0.55" material="pants"/>
    <geom type="capsule" fromto="0.02 0.09 0.52 0.42 0.09 0.52" size="0.065" material="pants"/>
    <geom type="capsule" fromto="0.02 -0.09 0.52 0.42 -0.09 0.52" size="0.065" material="pants"/>
    <geom type="capsule" fromto="0.44 0.09 0.5 0.46 0.09 0.08" size="0.05" material="pants"/>
    <geom type="capsule" fromto="0.44 -0.09 0.5 0.46 -0.09 0.08" size="0.05" material="pants"/>
    <geom type="box" size="0.11 0.045 0.035" pos="0.5 0.09 0.035" material="shoe"/>
    <geom type="box" size="0.11 0.045 0.035" pos="0.5 -0.09 0.035" material="shoe"/>
    <geom type="ellipsoid" size="0.1 0.165 0.17" pos="0 0 0.88" material="shirt"/>
    <geom type="ellipsoid" size="0.095 0.145 0.14" pos="0 0 0.7" material="shirt"/>
    <geom type="capsule" fromto="0 0 1.0 0 0 1.1" size="0.045" material="skin"/>
    <body name="{name}_head" pos="0.01 0 1.2">
      <geom type="ellipsoid" size="0.095 0.08 0.11" material="skin"/>
      <geom type="ellipsoid" size="0.1 0.088 0.07" pos="-0.005 0 0.045" material="cap"/>
      <geom type="sphere" size="0.012" pos="0.085 0.03 0.015" rgba="0.1 0.1 0.12 1"/>
      <geom type="sphere" size="0.012" pos="0.085 -0.03 0.015" rgba="0.1 0.1 0.12 1"/>
      <geom type="capsule" fromto="-0.08 0 -0.02 -0.2 0 -0.12" size="0.006" rgba="0.15 0.15 0.18 1"/>
        {eeg_cap()}
    </body>
    {'<geom type="box" size="0.06 0.12 0.16" pos="-0.17 0 0.85" material="motor"/><geom name="pack_led" type="box" size="0.002 0.08 0.01" pos="-0.11 0 0.95" rgba="0.3 0.9 0.5 1"/>' if exo else ''}
    {arm("L", exo)}
    {arm("R", exo)}
  </body>"""


ASSETS = """
  <texture name="sky" type="skybox" builtin="gradient" rgb1="0.93 0.94 0.95" rgb2="0.75 0.78 0.82" width="512" height="512"/>
  <texture name="floor" type="2d" builtin="checker" rgb1="0.78 0.76 0.72" rgb2="0.74 0.72 0.68" width="512" height="512" mark="edge" markrgb="0.7 0.68 0.64"/>
  <material name="floor" texture="floor" texrepeat="8 8" reflectance="0.15"/>
  <material name="wall" rgba="0.92 0.91 0.88 1"/>
  <material name="wall2" rgba="0.83 0.88 0.9 1"/>
  <material name="wood" rgba="0.62 0.47 0.33 1" specular="0.2"/>
  <material name="skin" rgba="0.80 0.62 0.50 1" specular="0.1"/>
  <material name="shirt" rgba="0.28 0.42 0.58 1"/>
  <material name="pants" rgba="0.22 0.24 0.28 1"/>
  <material name="shoe" rgba="0.12 0.12 0.13 1"/>
  <material name="cap" rgba="0.18 0.2 0.24 1" specular="0.4"/>
  <material name="carbon" rgba="0.13 0.14 0.16 1" specular="0.8" shininess="0.8" reflectance="0.1"/>
  <material name="alu" rgba="0.72 0.74 0.77 1" specular="1" shininess="0.9" reflectance="0.3"/>
  <material name="motor" rgba="0.95 0.55 0.22 1" specular="0.6" shininess="0.6"/>
  <material name="strap" rgba="0.2 0.22 0.25 1"/>
  <material name="steel" rgba="0.82 0.84 0.86 1" specular="1" shininess="0.9" reflectance="0.25"/>
  <material name="white" rgba="0.95 0.95 0.94 1" specular="0.3"/>
  <material name="chair" rgba="0.3 0.32 0.36 1"/>
"""

VISUAL = """
  <visual>
    <global offwidth="1920" offheight="1080"/>
    <quality shadowsize="8192" offsamples="8"/>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.25 0.25 0.25" specular="0.05 0.05 0.05"/>
    <map znear="0.01"/>
  </visual>"""


def chair(pos):
    return f"""
  <body pos="{pos}">
    <geom type="box" size="0.24 0.24 0.03" pos="0.05 0 0.45" material="chair"/>
    <geom type="box" size="0.03 0.24 0.3" pos="-0.2 0 0.78" material="chair"/>
    <geom type="cylinder" fromto="0.05 0 0.42 0.05 0 0.06" size="0.025" material="steel"/>
    <geom type="cylinder" fromto="0.05 0 0.05 0.05 0 0.04" size="0.25" material="chair"/>
  </body>"""
