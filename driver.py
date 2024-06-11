
import MatterSim
import time
import math
import cv2
import numpy as np
import os
import sys
import HA3DSim

TARGET_FPS = 20  # 目标帧率
FRAME_DURATION = 1.0 / TARGET_FPS  # 目标帧持续时间

def compute_fps(time_diff, rgb):
    # 计算FPS
    fps = 1 / time_diff if time_diff > 0 else 0
    # 更新上一帧的时间戳
    prev_frame_time = current_time
    # 将FPS值转换为字符串
    fps_text = f"FPS: {int(fps)}"
    # 在图像的左上角绘制FPS
    cv2.putText(rgb, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (255, 255, 255), 2, cv2.LINE_AA)
    # 如果处理得太快，等待剩余的时间
    if time_diff < FRAME_DURATION:
        time.sleep(FRAME_DURATION - time_diff)

dataset_path = os.path.join(os.environ.get("HA3D_SIMULATOR_DTAT_PATH"), "data/v1/scans")
WIDTH = 800
HEIGHT = 600
VFOV = math.radians(60)
HFOV = VFOV*WIDTH/HEIGHT
TEXT_COLOR = [230, 40, 40]

cv2.namedWindow('Python RGB')
cv2.namedWindow('Python Depth')


#sim = MatterSim.Simulator()
sim = HA3DSim.HASimulator()
#sim.setRenderingEnabled(False)
sim.setRealTimeRender(True)
sim.setDatasetPath(dataset_path)
sim.setCameraResolution(WIDTH, HEIGHT)
sim.setCameraVFOV(VFOV)
sim.setDepthEnabled(True) # Turn on depth only after running ./scripts/depth_to_skybox.py (see README.md)
sim.initialize()
#sim.newEpisode(['2t7WUuJeko7'], ['1e6b606b44df4a6086c0f97e826d4d15'], [0], [0])
scanId = '29hnd4uzFmX'
viewpointId = 'b14d29bea4b547d5923b3a09323b443d'
sim.newEpisode([scanId], [viewpointId], [0], [0])
#sim.newRandomEpisode(['1LXtFkjw3qL'])

heading = 0
elevation = 0
location = 0
ANGLEDELTA = 5 * math.pi / 180

print('\nPython Demo')
print('Use arrow keys to move the camera.')
print('Use number keys (not numpad) to move to nearby viewpoints indicated in the RGB view.')
print('Depth outputs are turned off by default - check driver.py:L20 to enable.\n')
# 初始化上一帧的时间戳
prev_frame_time = time.time()
while True:
    
    location = 0
    heading = 0
    elevation = 0

    state = sim.getState()[0]
    locations = state.navigableLocations
    rgb = np.array(state.rgb, copy=False)
    
    for idx, loc in enumerate(locations[1:]):
        # Draw actions on the screen
        fontScale = 3.0/loc.rel_distance
        x = int(WIDTH/2 + loc.rel_heading/HFOV*WIDTH)
        y = int(HEIGHT/2 - loc.rel_elevation/VFOV*HEIGHT)
        cv2.putText(rgb, str(idx + 1), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
            fontScale, TEXT_COLOR, thickness=3)
        # 获取当前时间
    current_time = time.time()
    # 计算两帧之间的时间差
    time_diff = current_time - prev_frame_time
    prev_frame_time = current_time
    compute_fps(time_diff, rgb)
    
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
        heading = -ANGLEDELTA
        sim.makeAction([location], [heading], [elevation])
    elif k == 82 or k == ord('w'):
        elevation = ANGLEDELTA
        sim.makeAction([location], [heading], [elevation])
    elif k == 83 or k == ord('d'):
        heading = ANGLEDELTA
        sim.makeAction([location], [heading], [elevation])
    elif k == 84 or k == ord('s'):
        elevation = -ANGLEDELTA
        sim.makeAction([location], [heading], [elevation])
    elif k == ord('c'):
        imgfile = f"{state.scanId}_{state.location.viewpointId}_{state.viewIndex}_{state.heading}_{state.elevation}"
        cv2.imwrite("sim_imgs/"+imgfile+"rgb.png", rgb)
        print(imgfile)


