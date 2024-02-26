import os
import trimesh
import imageio
import numpy as np
from src.render.rendermdm import get_renderer, render_frames
from src.utils.get_info import get_human_info, load_viewpointids
import MatterSim
import math
import requests
import argparse
import copy
from tqdm import tqdm
import csv
import json

class HC_Simulator(MatterSim.Simulator):
    def __init__(self,remote=False, ip="192.168.24.41", port="8080"):
        self.remote = remote
        self.address = f'http://{ip}:{port}'
        self.state_list = []
        self.state_index = -1
        self.scanId = 0
        self.viewpointId = 0
        self.WIDTH = 640
        self.HEIGHT = 480
        if self.remote:
            # 发送 POST 请求
            response = requests.post(self.address, json={'function': 'Simulator dynamicMatterSim'})
            print('POST response: ', response.text)
        else:
            super().__init__()
    
    def setCameraResolution(self, WIDTH, HEIGHT):
        if self.remote:
            # 发送 POST 请求
            response = requests.post(self.address, json={'function': 'Simulator setCameraResolution'
                                                         , 'width': WIDTH, 'height': HEIGHT})
            print('POST response: ', response.text)
        else:
            super().setCameraResolution(WIDTH, HEIGHT)

    def setCameraVFOV(self, VFOV):
        if self.remote:
            # 发送 POST 请求
            response = requests.post(self.address, json={'function': 'Simulator setCameraVFOV','VFOV': VFOV})
            print('POST response: ', response.text)
        else:
            super().setCameraVFOV(VFOV)

    def setDiscretizedViewingAngles(self, flag):
        if self.remote:
            # 发送 POST 请求
            response = requests.post(self.address, json={'function': 'Simulator setDiscretizedViewingAngles','flag': flag})
            print('POST response: ', response.text)
        else:
            super().setDiscretizedViewingAngles(flag)

    def setBatchSize(self, BatchSize):
        if self.remote:
            # 发送 POST 请求
            response = requests.post(self.address, json={'function': 'Simulator setBatchSize','BatchSize': BatchSize})
            print('POST response: ', response.text)
        else:
            super().setBatchSize(BatchSize)

    def initialize(self):
        if self.remote:
            # 发送 POST 请求
            response = requests.post(self.address, json={'function': 'Simulator initialize'})
            print('POST response: ', response.text)
        else:
            super().initialize()

    def newEpisode(self, scanId, viewpointId, heading, elevation):
        if self.remote:
            # 发送 POST 请求
            response = requests.post(self.address, json={'function': 'Simulator newEpisode',
                                                         "scanId":scanId,
                                                         "viewpointId":viewpointId,
                                                         "heading":heading,
                                                         "elevation":elevation})
            print('POST response: ', response.text)
        else:
            #super().newEpisode(scanId, viewpointId, heading, elevation)
            self.scanId = scanId[0]
            self.viewpointId = viewpointId[0]
            self.state_index += 1

    def makeAction(self, index, heading, elevation):
        if self.remote:
            # 发送 POST 请求
            response = requests.post(self.address, json={'function': 'Simulator makeAction',
                                                         "index":index,
                                                         "heading":heading,
                                                         "elevation":elevation})
            #print('POST response: ', response.text)
        else:
            #super().makeAction(index, heading, elevation)
            self.state_index += 1
    
    def getState(self, num_frames):
        if self.remote:
            # 发送 POST 请求
            response = requests.post(self.address, json={'function': 'Simulator getState'})
            #print('POST response: ', response.text)
            o_state = response.json()
            state = HC_SimState(o_state, remote=True)
            state.video = self.HCFusion(state, num_frames=num_frames)
        else:
            o_state = self.state_list[self.state_index]
            state = HC_SimState(o_state)
            assert self.scanId == state.scanId
            assert self.viewpointId == state.location.viewpointId
            state.video = self.HCFusion(state, num_frames=num_frames)
            
        return [state]


    def fill_image(self, img, num_frames=60):
        return np.array([img]*num_frames)

    def HCFusion(self, state, num_frames=60):
        location = state.location
        view_id = location.viewpointId
        human_angle, human_loc, motion_path = get_human_info(os.getenv('VLN_DATA_DIR'), state.scanId, view_id)
        background = state.rgb
        background_depth = np.squeeze(state.depth, axis=-1)
        #print(f"Background shape {background.shape}, background_depth shape {background_depth.shape}")

        if human_angle == None:
            imgs_ny = self.fill_image(background, num_frames)
            #print(imgs_ny.shape)
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
        #print(imgs_ny.shape)
        return imgs_ny
    
    def preRenderAll(self, VIEWPOINT_SIZE):
        # Loop all the viewpoints in the simulator
        TSV_FIELDNAMES = ["scanId", "step", "rgb", "depth", "location", "heading", "elevation", "viewIndex", "navigableLocations"]
        dir_path = f"{os.getenv('VLN_DATA_DIR')}/states"
        json_path = os.path.join(dir_path,"simulator_states.json")
        # 检查文件夹是否已经存在
        if not os.path.exists(dir_path):
            # 如果文件夹不存在，创建它
            os.makedirs(dir_path)
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                self.state_list = json.load(f)
            print("scan view state is exist")
            print(f"states num:{len(self.state_list)}")
            if len(self.state_list) < (len(viewpointIds)*VIEWPOINT_SIZE)
                return
        print("-------------------------Pre Renser all scan view------------------------")
        viewpointIds = load_viewpointids()
        bar = tqdm(viewpointIds)
        for i, (scanId, viewpointId) in enumerate(bar):
            bar.set_description()
            #bar = tqdm(range(VIEWPOINT_SIZE))
            for ix in range(VIEWPOINT_SIZE):
                #bar.set_description()
                if ix == 0:
                    super().newEpisode([scanId], [viewpointId], [0], [math.radians(-30)])
                elif ix % 12 == 0:
                    super().makeAction([0], [1.0], [1.0])
                else:
                    super().makeAction([0], [1.0], [0])

                state = super().getState()
                
                state_dic = state_to_dic(state[0])

                #save
                rgb_path = f"{state_dic['scanId']}_{state_dic['location']['viewpointId']}_{state_dic['step']}.png"
                depth_path = f"{state_dic['scanId']}_{state_dic['location']['viewpointId']}_{state_dic['step']}_depth.tiff"
                self.state_list.append(
                    {
                        "scanId" : state_dic["scanId"],
                        "step" : state_dic["step"], 
                        "rgb" : rgb_path, 
                        "depth" : depth_path, 
                        "location" : state_dic["location"], 
                        "heading" : state_dic["heading"], 
                        "elevation" : state_dic["elevation"], 
                        "viewIndex" : state_dic["viewIndex"], 
                        "navigableLocations" : state_dic["navigableLocations"]
                    }
                )
                if not os.path.exists(os.path.join(dir_path,rgb_path)):
                    imageio.imsave(os.path.join(dir_path,rgb_path), state_dic["rgb"])
                    imageio.imsave(os.path.join(dir_path,depth_path), state_dic["depth"])
                #print(self.state_list[0][0].scanId, self.state_list[0][0].location.viewpointId, self.state_list[0][0].step)
                #print(self.state_list[-1][0].scanId, self.state_list[-1][0].location.viewpointId, self.state_list[-1][0].step)
        #保存为JSON
        with open(json_path, 'w') as f:
            json.dump(self.state_list, f, indent=4)
        print(f"states num:{len(self.state_list)}")

class HC_SimState():
    def __init__(self,o_state):
        dir_path = f"{os.getenv('VLN_DATA_DIR')}/states"
        self.scanId = o_state["scanId"]
        self.step = o_state["step"]
        self.rgb = imageio.imread(os.path.join(dir_path, o_state["rgb"]))
        self.depth = imageio.imread(os.path.join(dir_path, o_state["depth"]))
        self.location = Location(o_state["location"])
        self.heading = o_state["heading"]
        self.elevation = o_state["elevation"]
        self.viewIndex = o_state["viewIndex"]
        self.navigableLocations = self.navigableLocations_to_object(o_state["navigableLocations"]) 
        self.video = []

    def navigableLocations_to_object(self, navigableLocations):
        new_navigableLocations = []
        for i in range(len(navigableLocations)):
            new_navigableLocations.append(Location(navigableLocations[i]))
        return new_navigableLocations

class Location():
    def __init__(self, location):
        self.viewpointId = location["viewpointId"]
        self.ix = location["ix"]
        self.x = location["x"]
        self.y = location["y"]
        self.z = location["z"]
        self.rel_heading = location["rel_heading"]
        self.rel_elevation = location["rel_elevation"]
        self.rel_distance = location["rel_distance"]

def state_to_dic(state):
    dic = {}
    dic["scanId"] = state.scanId
    dic["step"] = state.step
    dic["rgb"] = np.array(state.rgb, copy=False)
    #dic["rgb"] = state.rgb
    #imageio.imwrite('output.png', np.array(state.rgb, copy=False))
    dic["depth"] = np.array(state.depth, copy=False)
    dic["depth"] = state.depth
    dic["location"] = location_type_dic(state.location)
    dic["heading"] = state.heading
    dic["elevation"] = state.elevation
    dic["viewIndex"] = state.viewIndex
    dic["navigableLocations"] =  navigableLocations_type_dic(state.navigableLocations)
    return dic

def location_type_dic(location):
    dic = {            
        "viewpointId" : location.viewpointId,  
        "ix" : location.ix,                                            
        "x" : location.x,                                 
        "y" : location.y,
        "z" : location.z,
        "rel_heading" : location.rel_heading,                                  
        "rel_elevation" : location.rel_elevation,
        "rel_distance" : location.rel_distance
    }
    return dic

def navigableLocations_type_dic(navigableLocations):
    new_navigableLocations = []
    for i in range(len(navigableLocations)):
        new_navigableLocations.append(location_type_dic(navigableLocations[i]))
    return new_navigableLocations    

def main(args):
    WIDTH = 640
    HEIGHT = 480
    VFOV = math.radians(60)

    sim = HC_Simulator(remote=True, ip=args.ip, port=args.port)
    #sim.setDatasetPath(os.environ.get("MATTERPORT_DATA_DIR"))
    sim.setCameraResolution(WIDTH, HEIGHT)
    sim.setCameraVFOV(VFOV)
    #sim.setDepthEnabled(True) # Turn on depth only after running ./scripts/depth_to_skybox.py (see README.md)
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip', default='192.168.24.41')
    parser.add_argument('--port', default='8080')
    args = parser.parse_args()
    main(args)


