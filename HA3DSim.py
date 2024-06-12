import MatterSim
import os
import imageio
import numpy as np
from src.utils.get_info import getHumanOfScan, relHumanAngle, getAllHumanLocations
import math
import argparse
import pickle

# Path to HA3D simulator
HA3D_SIMULATOR_PATH = os.environ.get("HA3D_SIMULATOR_PATH")

def receiveMessage(pipeR2S):
    """Receive message from a named pipe and deserialize the data."""
    with open(pipeR2S, 'rb') as pipe_r2s:
        while True:
            # Read serialized data from pipe
            serialized_data = pipe_r2s.read()
            # If data is read, deserialize it
            if serialized_data:
                message = pickle.loads(serialized_data)['message']
                break

class HASimulator(MatterSim.Simulator):
    def __init__(self, pipeID=0):
        """Initialize the simulator with default settings and pipe IDs."""
        self.isRealTimeRender = False
        self.state = None
        self.allHumanLocations = {}
        self.states = []
        self.stateIndex = -1
        self.scanId = 0
        self.viewpointId = 0
        self.WIDTH = 640
        self.HEIGHT = 480
        self.VFOV = math.radians(60)
        self.pipeS2R = os.path.join(HA3D_SIMULATOR_PATH, f'pipe/my_S2R_pipe{pipeID}')
        self.pipeR2S = os.path.join(HA3D_SIMULATOR_PATH, f'pipe/my_R2S_pipe{pipeID}')
        print(f"Simulator PIPE {pipeID}")
        self.frameNum = 0
        self.framesPerStep = 16
        super().__init__()

    def setRenderingEnabled(self, isRealTimeRender):
        """Enable or disable real-time rendering mode."""
        self.isRealTimeRender = isRealTimeRender
        if isRealTimeRender:
            print("Real Time Rendering mode enabled!")
        else:
            scanIDs = []  # Example: ['2n8kARJN3HM', '1LXtFkjw3qL']
            self.allHumanLocations = getAllHumanLocations(scanIDs)
        super().setRenderingEnabled(isRealTimeRender)

    def setCameraResolution(self, WIDTH, HEIGHT):
        """Set the camera resolution for the simulator."""
        self.WIDTH = WIDTH
        self.HEIGHT = HEIGHT
        super().setCameraResolution(WIDTH, HEIGHT)

    def setCameraVFOV(self, VFOV):
        """Set the vertical field of view (VFOV) for the camera."""
        self.VFOV = VFOV
        super().setCameraVFOV(VFOV)

    def initialize(self):
        """Initialize the simulator and create renderer if real-time rendering is enabled."""
        super().initialize()
        if self.isRealTimeRender:
            data = {
                'function': 'create renderer',
                'WIDTH': self.WIDTH,
                'HEIGHT': self.HEIGHT
            }
            with open(self.pipeS2R, 'wb') as pipe:
                serialized_data = pickle.dumps(data)
                pipe.write(serialized_data)
            receiveMessage(self.pipeR2S)

    def newEpisode(self, scanId, viewpointId, heading, elevation):
        """Start a new episode with given parameters."""
        super().newEpisode(scanId, viewpointId, heading, elevation)
        if self.isRealTimeRender:
            self.state = super().getState()[0]
            if self.scanId != scanId:
                self.scanId = scanId
                human_list = getHumanOfScan(scanId[0])
                data = {
                    'function': 'set human',
                    'human_list': human_list,
                    'scanID': self.scanId
                }
                with open(self.pipeS2R, 'wb') as pipe:
                    serialized_data = pickle.dumps(data)
                    pipe.write(serialized_data)
                receiveMessage(self.pipeR2S)
            
            data = {
                'function': 'set agent',
                'VFOV': self.VFOV,
                'location': [self.state.location.x, self.state.location.y, self.state.location.z],
                'heading': self.state.heading,
                'elevation': self.state.elevation
            }
            with open(self.pipeS2R, 'wb') as pipe:
                serialized_data = pickle.dumps(data)
                pipe.write(serialized_data)
            receiveMessage(self.pipeR2S)
            self.renderScene()

    def makeAction(self, index, heading, elevation):
        """Perform an action in the simulation and update the state."""
        self.frameNum += self.framesPerStep
        if self.frameNum >= 120:
            self.frameNum = 0
        super().makeAction(index, heading, elevation)
        if self.isRealTimeRender:
            self.state = super().getState()[0]
            data = {
                'function': 'move agent',
                'VFOV': self.VFOV,
                'location': [self.state.location.x, self.state.location.y, self.state.location.z],
                'heading': self.state.heading,
                'elevation': self.state.elevation
            }
            with open(self.pipeS2R, 'wb') as pipe:
                serialized_data = pickle.dumps(data)
                pipe.write(serialized_data)
            receiveMessage(self.pipeR2S)
            self.renderScene()

    def renderScene(self):
        """Render the current scene in the simulator."""
        self.state = HASimState(self.state)
        self.background = self.state.rgb.astype(np.uint8)
        self.background_depth = np.squeeze(self.state.depth, axis=-1)
        data = {
            'function': 'render scene',
            'background': self.background,
            'background_depth': self.background_depth,
        }
        with open(self.pipeS2R, 'wb') as pipe:
            serialized_data = pickle.dumps(data)
            pipe.write(serialized_data)
        receiveMessage(self.pipeR2S)

    def getState(self, framesPerStep=1):
        """Get the current state of the simulator."""
        states = []
        self.framesPerStep = framesPerStep
        self.frameNum += self.framesPerStep
        if self.frameNum >= 120:
            self.frameNum = 0
        if self.isRealTimeRender:
            data = {
                'function': 'get state',
                'frame_num': self.frameNum
            }
            with open(self.pipeS2R, 'wb') as pipe_s2r:
                serialized_data = pickle.dumps(data)
                pipe_s2r.write(serialized_data)
            with open(self.pipeR2S, 'rb') as pipe_r2s:
                while True:
                    serialized_data = pipe_r2s.read()
                    if serialized_data:
                        data = pickle.loads(serialized_data)
                        self.state.rgb = data['rgb']
                        break
            self.state.humanState = self.getHumanState(self.state.scanId)
            states.append(self.state)
        else:
            for state in super().getState():
                state = HASimState(state)
                humanLocations = self.getHumanState(state.scanId)
                relHeading, relElevation, minDistance = relHumanAngle(
                    humanLocations,
                    [state.location.x, state.location.y, state.location.z],
                    state.heading,
                    state.elevation
                )
                state.isCrushed = int(minDistance <= 1)
                states.append(state)
        return states

    def getHumanState(self, scanID=''):
        """Get the human state for a given scan ID."""
        humanStates = []
        if self.isRealTimeRender:
            data = {
                'function': 'get human state',
                'frame_num': self.frameNum
            }
            with open(self.pipeS2R, 'wb') as pipe_s2r:
                serialized_data = pickle.dumps(data)
                pipe_s2r.write(serialized_data)
            with open(self.pipeR2S, 'rb') as pipe_r2s:
                while True:
                    serialized_data = pipe_r2s.read()
                    if serialized_data:
                        data = pickle.loads(serialized_data)
                        humanStates = data['human_state']
                        break
        else:
            for hs in self.allHumanLocations[scanID]:
                humanStates.append(hs[self.frameNum])
        return humanStates  

    def getStepState(self, frames=16, gap=4):
        """Get the state of the agent over multiple frames."""
        agentViewFrames = []
        for i in range(int(frames / gap)):
            self.makeAction([0], [0], [0])
            state = self.getState(framesPerStep=gap)[0]
            agentViewFrames.append(state.rgb)
        return state, np.array(agentViewFrames, copy=False)

class HASimState:
    def __init__(self, o_state):
        """Initialize the simulator state with given parameters."""
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

    def navigableLocationsToObject(self, navigableLocations):
        """Convert navigable locations to objects."""
        return [Location(loc) for loc in navigableLocations]

class Location:
    def __init__(self, location):
        """Initialize location with given parameters."""
        self.viewpointId = location["viewpointId"]
        self.ix = location["ix"]
        self.x = location["x"]
        self.y = location["y"]
        self.z = location["z"]
        self.relHeading = location["rel_heading"]
        self.relElevation = location["rel_elevation"]
        self.relDistance = location["rel_distance"]

def stateToDic(state):
    """Convert state to a dictionary."""
    dic = {
        "scanId": state.scanId,
        "step": state.step,
        "rgb": np.array(state.rgb, copy=False),
        "depth": np.array(state.depth, copy=False),
        "location": locationTypeDic(state.location),
        "heading": state.heading,
        "elevation": state.elevation,
        "viewIndex": state.viewIndex,
        "navigableLocations": navigableLocationsTypeDic(state.navigableLocations)
    }
    return dic

def locationTypeDic(location):
    """Convert location to a dictionary."""
    return {
        "viewpointId": location.viewpointId,
        "ix": location.ix,
        "x": location.x,
        "y": location.y,
        "z": location.z,
        "rel_heading": location.relHeading,
        "rel_elevation": location.relElevation,
        "rel_distance": location.relDistance
    }

def navigableLocationsTypeDic(navigableLocations):
    """Convert list of navigable locations to a list of dictionaries."""
    return [locationTypeDic(loc) for loc in navigableLocations]

def saveVideoBGR(frames, filename, fps):
    """Save a video from a list of frames."""
    writer = imageio.get_writer(filename, fps=fps)
    for frame in frames:
        writer.append_data(frame)
    writer.close()

def main(args):
    """Main function to set up and run the simulator."""
    WIDTH = 800
    HEIGHT = 600
    VFOV = math.radians(60)
    batch_size = 1
    dataset_path = os.path.join(os.environ.get("HA3D_SIMULATOR_DATA_PATH"), "data/v1/scans")
    sim = HASimulator()
    sim.setRenderingEnabled(True)
    sim.setBatchSize(batch_size)
    sim.setDatasetPath(dataset_path)
    sim.setCameraResolution(WIDTH, HEIGHT)
    sim.setCameraVFOV(VFOV)
    sim.setDepthEnabled(True)  # Turn on depth only after running ./scripts/depth_to_skybox.py
    sim.initialize()
    sim.newEpisode(['1LXtFkjw3qL'], ['0b22fa63d0f54a529c525afbf2e8bb25'], [0], [0])

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
        cv2.imwrite("sim_test.png", state.rgb)
        print(f"scanID: {state.scanId}")
        print(f"agent step {state.step}")
        for humanLocation in state.humanState:
            print(f"Human Location: {humanLocation}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
