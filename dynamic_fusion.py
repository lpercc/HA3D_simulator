import json
import os
import trimesh
import imageio
import numpy as np
from src.render.rendermdm import get_renderer, render_frames
import MatterSim
import time
import math
import cv2

def get_human_info(scan_id, agent_view_id):
    motion_dir = os.path.join(basic_data_dir,"human_motion_meshes")
        # 一共90个建筑场景数据
    with open('human_motion_text.json', 'r') as f:
        human_view_data = json.load(f)
        # 遍历建筑场景中每个人物视点，即人物所在位置的视点
            # 获取建筑场景所有视点信息（视点之间的关系）
    with open('con/pos_info/{}_pos_info.json'.format(scan_id), 'r') as f:
        pos_data = json.load(f)
        #print(len(pos_data))
    with open('con/con_info/{}_con_info.json'.format(scan_id), 'r') as f:
        connection_data = json.load(f)
    for human_view_id in human_view_data[scan_id]:
        # 人物视点编号
        human_motion = human_view_data[scan_id][human_view_id][0]
        human_model_id = human_view_data[scan_id][human_view_id][1]
        human_heading = human_view_data[scan_id][human_view_id][2]

        try:
            # 判断该视点是否可见目标视点（人物）
            if human_view_id == agent_view_id:
                connection_data[agent_view_id]["visible"].append(agent_view_id)
                print(f"human_view_id:{agent_view_id}")
            if human_view_id in connection_data[agent_view_id]['visible']:
                motion_path = os.path.join(motion_dir, human_motion.replace(' ', '_').replace('/', '_'), f"{human_model_id}_obj")
                human_loc = [pos_data[human_view_id][0], pos_data[human_view_id][1], pos_data[human_view_id][2]]
                return human_heading, human_loc, motion_path
        except KeyError:
            pass
     
def HCFusion(state, num_frames=60):
    location = state.location
    view_id = location.viewpointId
    human_angle, human_loc, motion_path = get_human_info(state.scanId, view_id)
    meshes = []
    # 从.obj文件创建mesh
    # 获取目录下的所有.obj文件，并按照序号从大到小排序
    obj_files = [f for f in os.listdir(motion_path) if f.endswith('.obj')]
    #print(obj_files[0].split('frame')[1].split('.obj')[0])
    sorted_obj_files = sorted(obj_files)
    for obj_file in sorted_obj_files[:num_frames]:
        obj_file.split('.')
        obj_path = os.path.join(motion_path,obj_file)
        mesh = trimesh.load(obj_path)
        meshes.append(mesh)

    background = np.array(state.rgb, copy=False)
    background_depth = np.squeeze(np.array(state.depth, copy=False), axis=-1)
    print(f"Background shape {background.shape}, background_depth shape {background_depth.shape}")

    cam_loc = [location.x, location.y, location.z]
    cam_angle = state.heading
    cam_elevation = state.elevation
    
    width = background.shape[1]
    height = background.shape[0]
    renderer = get_renderer(width, height)
    imgs = render_frames(meshes, background, background_depth,cam_loc, cam_angle, cam_elevation, human_loc, human_angle, renderer, view_id,color=[0, 0.8, 0.5])
    imgs_ny = np.array(imgs, copy=False)
    print(imgs_ny.shape)
    return imgs_ny

def main():
    WIDTH = 800
    HEIGHT = 600
    VFOV = math.radians(60)

    sim = MatterSim.Simulator()
    sim.setDatasetPath(os.environ.get("MATTERPORT_DATA_DIR"))
    sim.setCameraResolution(WIDTH, HEIGHT)
    sim.setCameraVFOV(VFOV)
    sim.setDepthEnabled(True) # Turn on depth only after running ./scripts/depth_to_skybox.py (see README.md)
    sim.initialize()
    #sim.newEpisode(['2t7WUuJeko7'], ['1e6b606b44df4a6086c0f97e826d4d15'], [0], [0])
    #sim.newEpisode(['1LXtFkjw3qL'], ['0b22fa63d0f54a529c525afbf2e8bb25'], [0], [0])
    sim.newEpisode(['1LXtFkjw3qL'], ["0b22fa63d0f54a529c525afbf2e8bb25"], [0], [0])

    heading = math.radians(-90)
    elevation = math.radians(-30)
    location = 0

    print('\nPython Demo')
    print('Use arrow keys to move the camera.')
    print('Use number keys (not numpad) to move to nearby viewpoints indicated in the RGB view.')
    print('Depth outputs are turned off by default - check driver.py:L20 to enable.\n')

    sim.makeAction([location], [heading], [elevation])

    state = sim.getState()[0]
    print(state.heading, state.elevation, state.viewIndex)
    frames = HCFusion(state, num_frames=60)
    writer = imageio.get_writer("test.mp4", fps=20)
    for frame in frames:
        writer.append_data(frame)
    writer.close()

if __name__ == "__main__":
    basic_data_dir = os.getenv('VLN_DATA_DIR')
    main()

