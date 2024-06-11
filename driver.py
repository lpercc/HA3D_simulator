import HA3DSim
import os
import numpy as np
import time
import math
import cv2
from multiprocessing import Process

TARGET_FPS = 20  # Target frames per second
FRAME_DURATION = 1.0 / TARGET_FPS  # Target frame duration

def computeFps(timeDiff, rgb):
    """
    Compute and display FPS on the image.

    Parameters:
    - timeDiff: Time difference between frames
    - rgb: RGB image
    """
    # Compute FPS
    fps = 1 / timeDiff if timeDiff > 0 else 0
    # Convert FPS value to string
    fpsText = f"FPS: {int(fps)}"
    # Draw FPS value on the top-left corner of the image
    cv2.putText(rgb, fpsText, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (255, 255, 255), 2, cv2.LINE_AA)
    # If processing is too fast, wait for the remaining time
    if timeDiff < FRAME_DURATION:
        time.sleep(FRAME_DURATION - timeDiff)

def runProgram(command, suppressOutput=False):
    """
    Run a program as a separate process.

    Parameters:
    - command: Command to run
    - suppressOutput: Whether to suppress the output
    """
    if suppressOutput:
        command += " >/dev/null 2>&1"
    print(command)
    os.system(f'python {command}')

# Set up simulator parameters
datasetPath = os.path.join(os.environ.get("HA3D_SIMULATOR_DATA_PATH"), "data/v1/scans")
WIDTH = 800
HEIGHT = 600
VFOV = math.radians(120)
HFOV = VFOV * WIDTH / HEIGHT
TEXT_COLOR = [230, 40, 40]

cv2.namedWindow('Python RGB')
cv2.namedWindow('Python Depth')

pipeId = 0
# Create child processes
Process(target=runProgram, args=(f"HA3DRender.py --pipeID {pipeId}", False)).start()

# Initialize simulator
sim = HA3DSim.HASimulator(pipeId)
sim.setRenderingEnabled(True)
sim.setDatasetPath(datasetPath)
sim.setCameraResolution(WIDTH, HEIGHT)
sim.setCameraVFOV(VFOV)
sim.setDepthEnabled(True)  # Turn on depth only after running ./scripts/depth_to_skybox.py (see README.md)
sim.initialize()
scanId = '17DRP5sb8fy'
viewpointId = '85c23efeaecd4d43a7dcd5b90137179e'
sim.newEpisode([scanId], [viewpointId], [0], [0])

heading = 0
elevation = 0
location = 0
ANGLE_DELTA = 5 * math.pi / 180

print('\nPython Demo')
print('Use arrow keys to move the camera.')
print('Use number keys (not numpad) to move to nearby viewpoints indicated in the RGB view.')
print('Depth outputs are turned off by default - check driver.py:L20 to enable.\n')

# Initialize the timestamp of the previous frame
prevFrameTime = time.time()

while True:
    location = 0
    heading = 0
    elevation = 0
    state = sim.getState()[0]
    locations = state.navigableLocations
    rgb = np.array(state.rgb, copy=False)

    for idx, loc in enumerate(locations[1:]):
        # Draw actions on the screen
        fontScale = 3.0 / loc.rel_distance
        x = int(WIDTH / 2 + loc.rel_heading / HFOV * WIDTH)
        y = int(HEIGHT / 2 - loc.rel_elevation / VFOV * HEIGHT)
        cv2.putText(rgb, str(idx + 1), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale, TEXT_COLOR, thickness=3)

    # Get the current time
    currentTime = time.time()
    # Compute the time difference between frames
    timeDiff = currentTime - prevFrameTime
    prevFrameTime = currentTime
    computeFps(timeDiff, rgb)

    cv2.imshow('Python RGB', rgb)

    depth = np.array(state.depth, copy=False)
    cv2.imshow('Python Depth', depth)
    k = cv2.waitKey(1)

    if k == -1:
        continue
    else:
        k = (k & 255)
    if k == ord('q'):
        break
    elif ord('1') <= k <= ord('9'):
        location = k - ord('0')
        if location >= len(locations):
            location = 0
        sim.makeAction([location], [heading], [elevation])
    elif k == 81 or k == ord('a'):
        heading = -ANGLE_DELTA
        sim.makeAction([location], [heading], [elevation])
    elif k == 82 or k == ord('w'):
        elevation = ANGLE_DELTA
        sim.makeAction([location], [heading], [elevation])
    elif k == 83 or k == ord('d'):
        heading = ANGLE_DELTA
        sim.makeAction([location], [heading], [elevation])
    elif k == 84 or k == ord('s'):
        elevation = -ANGLE_DELTA
        sim.makeAction([location], [heading], [elevation])
    elif k == ord('c'):
        imgfile = f"{state.scanId}_{state.location.viewpointId}_{state.viewIndex}_{state.heading}_{state.elevation}"
        cv2.imwrite("sim_imgs/" + imgfile + "rgb.png", rgb)
        print(imgfile)

cv2.destroyAllWindows()
