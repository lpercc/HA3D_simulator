import os
import trimesh
import imageio
import numpy as np
from src.render.rendermdm import get_renderer, render_frames
from src.utils.get_info import get_human_info
import MatterSim
import math

class HC_Simulator(MatterSim.Simulator):
    def __init__(self):
        super().__init__()
    
    def fill_image(self, img, num_frames=60):
        return np.array([img]*num_frames)

    def HCFusion(self, state, num_frames=60):
        location = state.location
        view_id = location.viewpointId
        human_angle, human_loc, motion_path = get_human_info(os.getenv('VLN_DATA_DIR'), state.scanId, view_id)
        background = np.array(state.rgb, copy=False)
        background_depth = np.squeeze(np.array(state.depth, copy=False), axis=-1)
        print(f"Background shape {background.shape}, background_depth shape {background_depth.shape}")

        if human_angle == None:
            imgs_ny = self.fill_image(background, num_frames)
            print(imgs_ny.shape)
            return imgs_ny
        
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
    
    def getState(self, num_frames):
        o_state = super().getState()[0]
        state = HC_SimState(o_state)
        state.video = self.HCFusion(o_state, num_frames=num_frames)
        return [state]

class HC_SimState():
    def __init__(self,o_state):
        self.scanId = o_state.scanId
        self.step = o_state.step
        self.rgb = o_state.rgb
        self.depth = o_state.depth
        self.location = o_state.location
        self.heading = o_state.heading
        self.elevation = o_state.elevation
        self.viewIndex = o_state.viewIndex
        self.navigableLocations = o_state.navigableLocations
        self.video = []
def main():
    WIDTH = 640
    HEIGHT = 480
    VFOV = math.radians(60)

    sim = HC_Simulator()
    sim.setDatasetPath(os.environ.get("MATTERPORT_DATA_DIR"))
    sim.setCameraResolution(WIDTH, HEIGHT)
    sim.setCameraVFOV(VFOV)
    sim.setDepthEnabled(True) # Turn on depth only after running ./scripts/depth_to_skybox.py (see README.md)
    sim.initialize()
    #sim.newEpisode(['2t7WUuJeko7'], ['1e6b606b44df4a6086c0f97e826d4d15'], [0], [0])
    #sim.newEpisode(['1LXtFkjw3qL'], ['0b22fa63d0f54a529c525afbf2e8bb25'], [0], [0])
    sim.newEpisode(['1LXtFkjw3qL'], ["0b302846f0994ec9851862b1d317d7f2"], [0], [0])

    heading = math.radians(-30)
    elevation = math.radians(-0)
    location = 0

    print('\nPython Demo')
    print('Use arrow keys to move the camera.')
    print('Use number keys (not numpad) to move to nearby viewpoints indicated in the RGB view.')
    print('Depth outputs are turned off by default - check driver.py:L20 to enable.\n')

    sim.makeAction([location], [heading], [elevation])
    state = sim.getState(num_frames=60)[0]
    
    print(state.heading, state.elevation, state.viewIndex)
    frames = state.video
    np.save("test_frames.npy", frames)
    writer = imageio.get_writer("test.mp4", fps=20)
    for frame in frames:
        writer.append_data(frame)
    writer.close()

if __name__ == "__main__":
    main()

