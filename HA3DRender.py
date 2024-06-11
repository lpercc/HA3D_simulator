import os
import pickle
import numpy as np
from src.render.renderer import getRenderer
import argparse

# Set environment variable for OpenGL
os.environ['PYOPENGL_PLATFORM'] = 'egl'

def sendMessage(pipeR2SPath, message):
    """
    Send a message through the specified pipe.
    
    Parameters:
    - pipeR2SPath: Path to the pipe for sending messages
    - message: Message to be sent
    """
    with open(pipeR2SPath, 'wb') as pipeR2S:
        data = {'message': message}
        serializedData = pickle.dumps(data)
        pipeR2S.write(serializedData)

def main(args):
    pipeS2RPath = f'./pipe/my_S2R_pipe{args.pipeID}'
    pipeR2SPath = f'./pipe/my_R2S_pipe{args.pipeID}'

    # Check if pipe files exist, if not, create them
    if not os.path.exists(pipeS2RPath):
        os.mkfifo(pipeS2RPath)
        print(f'Created pipe: {pipeS2RPath}')

    if not os.path.exists(pipeR2SPath):
        os.mkfifo(pipeR2SPath)
        print(f'Created pipe: {pipeR2SPath}')

    with open(pipeS2RPath, 'rb') as pipeS2R:
        while True:
            # Read serialized data from pipe
            serializedData = pipeS2R.read()
            if serializedData:
                # Deserialize the data
                data = pickle.loads(serializedData)
                function = data['function']

                if function == 'create renderer':
                    width = data['WIDTH']
                    height = data['HEIGHT']
                    renderer = getRenderer(width, height)
                    message = f"SUCCESS {function}: WIDTH: {width}, HEIGHT: {height}."
                    print(message)
                    sendMessage(pipeR2SPath, message)

                elif function == 'set human':
                    humanList = data['human_list']
                    renderer.newHumans(humanList)
                    message = f"SUCCESS {function}: {len(humanList)} humans of Scan {data['scanID']}."
                    print(message)
                    sendMessage(pipeR2SPath, message)

                elif function == 'set agent':
                    vfov = data['VFOV']
                    location = data['location']
                    heading = data['heading']
                    elevation = data['elevation']
                    renderer.newAgent(vfov, location, heading, elevation)
                    message = (f"SUCCESS {function}: VFOV: {vfov}, location: ({location[0]}, {location[1]}, {location[2]}), "
                               f"heading: {heading}, elevation: {elevation}")
                    print(message)
                    sendMessage(pipeR2SPath, message)

                elif function == 'move agent':
                    vfov = data['VFOV']
                    location = data['location']
                    heading = data['heading']
                    elevation = data['elevation']
                    renderer.moveAgent(vfov, location, heading, elevation)
                    message = (f"SUCCESS {function}: VFOV: {vfov}, location: ({location[0]}, {location[1]}, {location[2]}), "
                               f"heading: {heading}, elevation: {elevation}")
                    print(message)
                    sendMessage(pipeR2SPath, message)

                elif function == 'render scene':
                    background = data['background']
                    backgroundDepth = data['background_depth']
                    message = f"SUCCESS {function}: {np.sum(background)}, {background.shape}"
                    print(message)
                    sendMessage(pipeR2SPath, message)

                elif function == 'get state':
                    frameNum = data['frame_num']
                    rgb, _ = renderer.renderAgent(frameNum, background, backgroundDepth)
                    message = f"SUCCESS {function}: frame_num {frameNum}"
                    # Send the state data back
                    with open(pipeR2SPath, 'wb') as pipeR2S:
                        stateData = {
                            'function': 'get state',
                            'frame_num': frameNum,
                            'rgb': rgb
                        }
                        serializedData = pickle.dumps(stateData)
                        pipeR2S.write(serializedData)

                elif function == 'get human state':
                    frameNum = data['frame_num']
                    humanLoc = renderer.getHumanLocation(frameNum)
                    message = f"SUCCESS {function}: frame_num {frameNum}"
                    with open(pipeR2SPath, 'wb') as pipeR2S:
                        humanStateData = {
                            'function': 'get human state',
                            'frame_num': frameNum,
                            'human_state': humanLoc
                        }
                        serializedData = pickle.dumps(humanStateData)
                        pipeR2S.write(serializedData)

                else:
                    print(data)

    print("Render Process Message: FINISH")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pipeID', type=int, required=True, help='ID for the pipe to be created')
    args = parser.parse_args()
    main(args)
