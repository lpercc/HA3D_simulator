import sys
sys.path.append('build')

from MatterSim import Simulator
import math
import cv2
import json
import numpy as np
import os


sim = Simulator()
sim.setDatasetPath(os.environ.get("HC3D_SIMULATOR_DTAT_PATH"))
sim.setCameraResolution(500, 300)
sim.setCameraVFOV(math.radians(60))
sim.setElevationLimits(math.radians(-40),math.radians(50))
sim.initialize()

with open("src/test/rendertest_spec.json") as f:
    spec = json.load(f)
    for tc in spec[:1]:
        tc = tc[0]
        sim.newEpisode([tc["scanId"]], [tc["viewpointId"]], [tc["heading"]], [tc["elevation"]])
        state = sim.getState()[0]
        im = np.array(state.rgb, copy=False)
        imgfile = tc["reference_image"]
        cv2.imwrite("sim_imgs/"+imgfile, im);
        #cv2.imshow('rendering', im)
        #cv2.waitKey(0)


