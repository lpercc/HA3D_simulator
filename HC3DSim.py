import os
import trimesh
import imageio
import cv2
import numpy as np
from src.utils.get_info import get_human_info, load_viewpointids, getAllHuman, print_file_and_line_quick
import MatterSim
import math
import argparse
import copy
from tqdm import tqdm
import json
import pickle

def receiveMessage(pipe_R2S):
    with open(pipe_R2S, 'rb') as pipe_r2s:
        while True:
            # 读取序列化的数据
            serialized_data = pipe_r2s.read()
            # 如果读取到数据，反序列化
            if serialized_data:
                message = pickle.loads(serialized_data)['message']
                print(message)
                break


class HCSimulator(MatterSim.Simulator):
    def __init__(self):
        self.isRealTimeRender = False
        self.state = None
        self.states = [self.state]
        self.state_list = []
        self.state_index = -1
        self.scanId = 0
        self.viewpointId = 0
        self.WIDTH = 640
        self.HEIGHT = 480
        self.VFOV = math.radians(60)
        self.pipe_S2R = './pipe/my_S2R_pipe'
        self.pipe_R2S = './pipe/my_R2S_pipe'
        self.frame_num = 0
        super().__init__()
    
    

    def setCameraResolution(self, WIDTH, HEIGHT):
        self.WIDTH = WIDTH
        self.HEIGHT = HEIGHT
        super().setCameraResolution(WIDTH, HEIGHT)

    def setCameraVFOV(self, VFOV):
        self.VFOV = VFOV
        super().setCameraVFOV(VFOV)

    def setDiscretizedViewingAngles(self, flag):
        super().setDiscretizedViewingAngles(flag)

    def setBatchSize(self, BatchSize):
        super().setBatchSize(BatchSize)

    def setRealTimeRender(self, isRealTimeRender):
        self.isRealTimeRender = isRealTimeRender
        print("Real Time Rendering mode!!!")

    def initialize(self, viewpoint_s=0):
        super().initialize()
        if not self.isRealTimeRender:
            self.state_index = self.state_index + viewpoint_s
        else:
            data = {
                'function':'create renderer',
                'WIDTH':self.WIDTH,
                'HEIGHT':self.HEIGHT
            }
            with open(self.pipe_S2R, 'wb') as pipe:
                # 序列化数据
                serialized_data = pickle.dumps(data)
                # 写入到命名管道
                pipe.write(serialized_data)
                print(f"Waiting {data['function']}")
            receiveMessage(self.pipe_R2S)

    def newEpisode(self, scanId, viewpointId, heading, elevation):
        if not self.isRealTimeRender:
            self.scanId = scanId[0]
            self.viewpointId = viewpointId[0]
            self.state_index += 1
        else:
            print("Loading episode ......")
            super().newEpisode(scanId, viewpointId, heading, elevation)
            self.state = super().getState()[0]
            if self.scanId != scanId:
                self.scanId = scanId
                human_list = getAllHuman(scanId[0])
                data = {
                    'function':'set human',
                    'human_list':human_list,
                }
                with open(self.pipe_S2R, 'wb') as pipe:
                    # 序列化数据
                    serialized_data = pickle.dumps(data)
                    # 写入到命名管道
                    pipe.write(serialized_data)
                    print(f"Waiting {data['function']}")
                receiveMessage(self.pipe_R2S)
            print("over Loading")
            data = {
                'function':'set agent',
                'VFOV':self.VFOV,
                'location':[self.state.location.x, self.state.location.y, self.state.location.z],
                'heading':self.state.heading,
                'elevation':self.state.elevation
            }
            with open(self.pipe_S2R, 'wb') as pipe:
                # 序列化数据
                serialized_data = pickle.dumps(data)
                # 写入到命名管道
                pipe.write(serialized_data)
                print(f"Waiting {data['function']}")
            receiveMessage(self.pipe_R2S)
            self.renderScene()

    def makeAction(self, index, heading, elevation):
        if not self.isRealTimeRender:
            self.index = index[0]
            self.state_index += 1
        else:
            super().makeAction(index, heading, elevation)
            self.state = super().getState()[0]
            data = {
                'function':'move agent',
                'VFOV':self.VFOV,
                'location':[self.state.location.x, self.state.location.y, self.state.location.z],
                'heading':self.state.heading,
                'elevation':self.state.elevation
            }
            with open(self.pipe_S2R, 'wb') as pipe:
                # 序列化数据
                serialized_data = pickle.dumps(data)
                # 写入到命名管道
                pipe.write(serialized_data)
                print(f"Waiting {data['function']}")
            receiveMessage(self.pipe_R2S)
            self.renderScene()
            

    def renderScene(self):
        self.state = HCSimState(self.state, self.isRealTimeRender)
        #self.background = cv2.cvtColor(self.state.rgb, cv2.COLOR_BGR2RGB).astype(np.uint8)
        self.background = self.state.rgb.astype(np.uint8)
        self.background_depth = np.squeeze(self.state.depth, axis=-1)
        data = {
            'function':'rendering scene',
            'background':self.background,
            'background_depth':self.background_depth,
        }
        with open(self.pipe_S2R, 'wb') as pipe:
            # 序列化数据
            serialized_data = pickle.dumps(data)
            # 写入到命名管道
            pipe.write(serialized_data)
            print(f"Waiting {data['function']}")
        receiveMessage(self.pipe_R2S)


    def getState(self):
        if not self.isRealTimeRender:
            o_state = self.state_list[self.state_index]
            state = HCSimState(o_state,self.isRealTimeRender)
            assert self.scanId == state.scanId
            assert self.viewpointId == state.location.viewpointId
        else:
            data = {
                'function':'get rgb',
                'frame_num':self.frame_num
            }
            with open(self.pipe_S2R, 'wb') as pipe_s2r:
                # 序列化数据
                serialized_data = pickle.dumps(data)
                # 写入到命名管道
                pipe_s2r.write(serialized_data)
                #print(f"Waiting {data['function']}, frame_num {data['frame_num']}")
            #receiveMessage(self.pipe_R2S)
            with open(self.pipe_R2S, 'rb') as pipe_r2s:
                while True:
                    # 读取序列化的数据
                    serialized_data = pipe_r2s.read()
                    # 如果读取到数据，反序列化
                    if serialized_data:
                        data = pickle.loads(serialized_data)
                        self.state.rgb = data['rgb']
                        #print(np.sum(self.state.rgb))
                        #print(f"SUCCESS {data['function']}, frame_num {data['frame_num']}")
                        break
            self.frame_num += 1
            if self.frame_num == 60:
                 self.frame_num = 0
        self.states[0] = self.state
        return self.states

    def preRenderAll(self, VIEWPOINT_SIZE):
        # Loop all the viewpoints in the simulator
        TSV_FIELDNAMES = ["scanId", "step", "rgb", "depth", "location", "heading", "elevation", "viewIndex", "navigableLocations"]
        dir_path = f"{os.getenv('VLN_DATA_DIR')}/states"
        json_path = os.path.join(dir_path,"simulator_states.json")
        viewpointIds = load_viewpointids()
        # 检查文件夹是否已经存在
        if not os.path.exists(dir_path):
            # 如果文件夹不存在，创建它
            os.makedirs(dir_path)
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                state_list = json.load(f)
            print(f"states num:{len(state_list)}")
            if len(state_list) == (len(viewpointIds)*VIEWPOINT_SIZE):
                #print("pass")
                #pass
                print("scan view state is exist")
                self.state_list = state_list
                return
        print("-------------------------Pre Renser all scan view------------------------")
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
                if np.sum(state_dic["rgb"]) == 0:
                    raise ValueError("Image is None!!!")
                if not (os.path.exists(os.path.join(dir_path,rgb_path)) and os.path.exists(os.path.join(dir_path,depth_path))) :
                    imageio.imsave(os.path.join(dir_path,rgb_path), state_dic["rgb"])
                    imageio.imsave(os.path.join(dir_path,depth_path), state_dic["depth"])
                #print(self.state_list[0][0].scanId, self.state_list[0][0].location.viewpointId, self.state_list[0][0].step)
                #print(self.state_list[-1][0].scanId, self.state_list[-1][0].location.viewpointId, self.state_list[-1][0].step)
        #保存为JSON
        with open(json_path, 'w') as f:
            json.dump(self.state_list, f, indent=4)
        print(f"states num:{len(self.state_list)}")

class HCSimState():
    def __init__(self,o_state,isRealTimeRender):

        if isRealTimeRender:
            self.scanId = o_state.scanId
            self.step = o_state.step
            self.rgb = np.array(o_state.rgb, copy=False)
            self.depth = np.array(o_state.depth, copy=False)
            self.location = o_state.location
            self.heading = o_state.heading
            self.elevation = o_state.elevation
            self.viewIndex = o_state.viewIndex
            self.navigableLocations = o_state.navigableLocations
        else:
            dir_path = f"{os.getenv('VLN_DATA_DIR')}/states"
            self.scanId = o_state["scanId"]
            self.step = o_state["step"]
            self.rgb = cv2.imread(os.path.join(dir_path, o_state["rgb"]))
            self.depth = cv2.imread(os.path.join(dir_path, o_state["depth"]))
            self.location = Location(o_state["location"])
            self.heading = o_state["heading"]
            self.elevation = o_state["elevation"]
            self.viewIndex = o_state["viewIndex"]
            self.navigableLocations = self.navigableLocations_to_object(o_state["navigableLocations"]) 

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

def save_video_bgr(frames, filename, fps):
    writer = imageio.get_writer(filename, fps=fps)
    for frame in frames:
        writer.append_data(frame)
    writer.close()

def main(args):
    WIDTH = 800
    HEIGHT = 600
    VFOV = math.radians(60)
    dataset_path = os.path.join(os.environ.get("HC3D_SIMULATOR_DTAT_PATH"), "data/v1/scans")
    sim = HCSimulator()
    sim.setRealTimeRender(True)
    sim.setDatasetPath(dataset_path)
    sim.setCameraResolution(WIDTH, HEIGHT)
    sim.setCameraVFOV(VFOV)
    sim.setDepthEnabled(True) # Turn on depth only after running ./scripts/depth_to_skybox.py (see README.md)
    sim.initialize()
    #sim.newEpisode(['2t7WUuJeko7'], ['1e6b606b44df4a6086c0f97e826d4d15'], [0], [0])
    #sim.newEpisode(['1LXtFkjw3qL'], ['0b22fa63d0f54a529c525afbf2e8bb25'], [0], [0])
    scanId = '2n8kARJN3HM'
    viewpointId = '840cd9be95274178b83c956386943c99'
    sim.newEpisode([scanId], [viewpointId], [0], [0])

    heading = 4.765598775598298
    elevation = 0
    location = 0

    print('\nPython Demo')
    print('Use arrow keys to move the camera.')
    print('Use number keys (not numpad) to move to nearby viewpoints indicated in the RGB view.')
    print('Depth outputs are turned off by default - check driver.py:L20 to enable.\n')

    sim.makeAction([location], [heading], [elevation])
    state = sim.getState(num_frames=60)[0]
    cv2.imwrite('test.jpg', state.rgb)
    print(state.heading, state.elevation, state.viewIndex)
    
    #frames = state.video
    #np.save("test_frames.npy", frames)
    #video_file = f"{state.scanId}_{state.location.viewpointId}_{state.viewIndex}_{state.heading}_{state.elevation}.mp4"
    #save_video_bgr(frames, video_file, 20)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)


