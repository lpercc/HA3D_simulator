

import MatterSim
import os
import imageio
import numpy as np
from src.utils.get_info import getHumanOfScan, relHumanAngle, getAllHumanLocations
import math
import argparse
import pickle

def receiveMessage(pipe_R2S):
    with open(pipe_R2S, 'rb') as pipe_r2s:
        while True:
            # 读取序列化的数据
            serialized_data = pipe_r2s.read()
            # 如果读取到数据，反序列化
            if serialized_data:
                message = pickle.loads(serialized_data)['message']
                #print(message)
                break


class HCSimulator(MatterSim.Simulator):
    def __init__(self, pipeID=0):
        self.isRealTimeRender = False
        self.state = None
        self.allHumanLocations = {}
        self.states = []
        self.state_index = -1
        self.scanId = 0
        self.viewpointId = 0
        self.WIDTH = 640
        self.HEIGHT = 480
        self.VFOV = math.radians(60)
        self.pipe_S2R = f'./pipe/my_S2R_pipe{pipeID}'
        self.pipe_R2S = f'./pipe/my_R2S_pipe{pipeID}'
        print(f"Simulator PIPE {pipeID}")
        self.frame_num = -1
        super().__init__()

    def setCameraResolution(self, WIDTH, HEIGHT):
        self.WIDTH = WIDTH
        self.HEIGHT = HEIGHT
        super().setCameraResolution(WIDTH, HEIGHT)

    def setCameraVFOV(self, VFOV):
        self.VFOV = VFOV
        super().setCameraVFOV(VFOV)

    def setRenderingEnabled(self, isRealTimeRender):
        self.isRealTimeRender = isRealTimeRender
        if isRealTimeRender:
            print("Real Time Rendering mode!!!")
        else:
            scanIDs = [] # like ['2n8kARJN3HM', '1LXtFkjw3qL']
            self.allHumanLocations = getAllHumanLocations(scanIDs)
        super().setRenderingEnabled(isRealTimeRender)

    def initialize(self):
        super().initialize()
        if self.isRealTimeRender:
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
                #print(f"Waiting {data['function']}")
            receiveMessage(self.pipe_R2S)

    def newEpisode(self, scanId, viewpointId, heading, elevation):
        # one building one batch
        super().newEpisode(scanId, viewpointId, heading, elevation)
        if self.isRealTimeRender:
            #print("Loading episode ......")
            self.state = super().getState()[0]
            if self.scanId != scanId:
                self.scanId = scanId
                human_list = getHumanOfScan(scanId[0])
                data = {
                    'function':'set human',
                    'human_list':human_list,
                    'scanID':self.scanId
                }
                with open(self.pipe_S2R, 'wb') as pipe:
                    # 序列化数据
                    serialized_data = pickle.dumps(data)
                    # 写入到命名管道
                    pipe.write(serialized_data)
                    #print(f"Waiting {data['function']}")
                receiveMessage(self.pipe_R2S)
            #print("over Loading")
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
                #print(f"Waiting {data['function']}")
            receiveMessage(self.pipe_R2S)
            self.renderScene()

    def makeAction(self, index, heading, elevation):
        super().makeAction(index, heading, elevation)
        if self.isRealTimeRender:
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
                #print(f"Waiting {data['function']}")
            receiveMessage(self.pipe_R2S)
            self.renderScene()

    def renderScene(self):
        self.state = HCSimState(self.state)
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
            #print(f"Waiting {data['function']}")
        receiveMessage(self.pipe_R2S)

    def getState(self, framesPerStep=1):
        states = []
        if self.isRealTimeRender:
            self.frame_num += framesPerStep
            data = {
                'function':'get state',
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
            self.state.humanState = self.getHumanState(self.state.scanId)
            states.append(self.state)
        
        else:
            self.frame_num += framesPerStep
            for state in super().getState():
                state = HCSimState(state)
                humanLocations = self.getHumanState(state.scanId)
                relHeading, relElevation, minDistance = relHumanAngle(humanLocations, 
                                                    [state.location.x, state.location.y, state.location.z], 
                                                    state.heading,
                                                    state.elevation)

                if minDistance <= 1:
                    state.isCrushed = 1
                else:
                    state.isCrushed = 0
                states.append(state)
        if self.frame_num >= 80:
            self.frame_num = 0

        return states

    def getHumanState(self, scanID=''):
        humanStates = []
        if self.isRealTimeRender:
            data = {
                'function':'get human state',
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
                        humanStates = data['human_state']
                        #print(np.sum(self.state.rgb))
                        #print(f"SUCCESS {data['function']}, frame_num {data['frame_num']}")
                        break
        else:
            for hs in self.allHumanLocations[scanID]:
                humanStates.append(hs[self.frame_num])
        return humanStates  

    def getStepState(self,framesPerStep=16,gap=4):
        agentViewFrames = []
        for i in range(int(framesPerStep/gap)):
            state = self.getState(framesPerStep=gap)[0]
            agentViewFrames.append(state.rgb)
        #state.humanState = self.getHumanState()
        return state, np.array(agentViewFrames, copy=False)


class HCSimState():
    def __init__(self,o_state):

        self.scanId = o_state.scanId
        self.step = o_state.step
        self.rgb = np.array(o_state.rgb, copy=False)
        self.depth = np.array(o_state.depth, copy=False)
        self.location = o_state.location
        self.heading = o_state.heading
        self.elevation = o_state.elevation
        self.viewIndex = o_state.viewIndex
        self.navigableLocations = o_state.navigableLocations
        self.isCrushed = 0

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
    batch_size = 1
    dataset_path = os.path.join(os.environ.get("HC3D_SIMULATOR_DTAT_PATH"), "data/v1/scans")
    sim = HCSimulator()
    sim.setRenderingEnabled(True)
    sim.setBatchSize(batch_size)
    sim.setDatasetPath(dataset_path)
    sim.setCameraResolution(WIDTH, HEIGHT)
    sim.setCameraVFOV(VFOV)
    sim.setDepthEnabled(True) # Turn on depth only after running ./scripts/depth_to_skybox.py (see README.md)
    sim.initialize()
    #sim.newEpisode(['2t7WUuJeko7'], ['1e6b606b44df4a6086c0f97e826d4d15'], [0], [0])
    sim.newEpisode(['1LXtFkjw3qL'], ['0b22fa63d0f54a529c525afbf2e8bb25'], [0], [0])
    #scanIds = ['2n8kARJN3HM', '1LXtFkjw3qL']
    #viewpointIds = ['840cd9be95274178b83c956386943c99', '0b22fa63d0f54a529c525afbf2e8bb25']
    #sim.newEpisode(scanIds[:batch_size], viewpointIds[:batch_size], [0]*batch_size, [0]*batch_size)

    heading = 0
    elevation = 0
    location = 0

    print('\nPython Demo')
    print('Use arrow keys to move the camera.')
    print('Use number keys (not numpad) to move to nearby viewpoints indicated in the RGB view.')
    print('Depth outputs are turned off by default - check driver.py:L20 to enable.\n')

    sim.makeAction([location], [heading], [elevation])
    states = sim.getState()
    for state in states:
        assert np.sum(state.rgb) > 0
        assert np.sum(state.depth) > 0
        import cv2
        cv2.imwrite("sim_test.png",state.rgb)
        print(f"scanID:{state.scanId}")
        print(f"agent step {state.step}")
        for humanLocation in state.humanState:
            print(f"Human Location:{humanLocation}")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)


