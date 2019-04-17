#!/usr/bin/env python
# demonstration of markers (visual-only geoms)

import math
import time
import os
import numpy as np
from mujoco_py import load_model_from_xml, MjSim, MjViewer

MODEL_XML = """
<?xml version="1.0" ?>
<mujoco>
    <worldbody>
        <body name="box" pos="0 0 0.2">
            <geom size="0.15 0.15 0.15" type="box"/>
            <joint axis="1 0 0" name="box:x" type="slide"/>
            <joint axis="0 1 0" name="box:y" type="slide"/>
        </body>
        <body name="floor" pos="0 0 0.025">
            <geom size="1.0 1.0 0.02" rgba="0 1 0 1" type="box"/>
        </body>
    </worldbody>
</mujoco>
"""

MODEL_XML_1 = """
<?xml version="1.0" ?>
<mujoco>
    <worldbody>
    <body>
      <geom name="finger"
            type="box"
            size="0.0125 0.0125 0.0125"
            condim="4"
            density="567">
      </geom>
      <geom name="finger_hidden"
            type="box"
            size="0.0125 0.0125 0.0125"
            condim="4"
            contype="0"
            conaffinity="0"
            mass="0">
      </geom>
      <site name="finger_center"
            pos="0.25 0 0.5">
      </site>
      <joint name="finger:joint"
             type="free"
             damping="0.01">
      </joint>
    </body>
    </worldbody>
</mujoco>
"""


model = load_model_from_xml(MODEL_XML)
sim = MjSim(model)
viewer = MjViewer(sim)
step = 0
while True:
    t = time.time()
    x, y = math.cos(t), math.sin(t)
    viewer.add_marker(pos=np.array([x, y, 1]),
                      label=str(t))
    viewer.render()

    step += 1
    if step > 100 and os.getenv('TESTING') is not None:
        break
