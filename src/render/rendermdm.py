import numpy as np
import imageio
import os
import argparse
from tqdm import tqdm
from .renderer import getRenderer
import trimesh
import cv2
from src.utils.concat_skybox import concat

DOWNSIZED_WIDTH = 512
DOWNSIZED_HEIGHT = 512

basicDataDir = os.environ.get("HA3D_SIMULATOR_DATA_PATH")

def getRotation(theta=np.pi):
    """
    Get a rotation matrix for a given angle.
    
    Parameters:
    - theta: Rotation angle in radians
    """
    import src.utils.rotation_conversions as geometry
    import torch
    axis = torch.tensor([0, 1, 0], dtype=torch.float)
    axisAngle = theta * axis
    matrix = geometry.axisAngleToMatrix(axisAngle)
    return matrix.numpy()

def renderVideo(meshes, background, camLoc, camAngle, humanLoc, humanAngle, renderer, outputVideoPath, viewId, scanId, humanViewId, color=[0, 0.8, 0.5]):
    """
    Render a video from the given meshes and background.
    
    Parameters:
    - meshes: List of meshes to render
    - background: Background image
    - camLoc: Camera location
    - camAngle: Camera angle
    - humanLoc: Human location
    - humanAngle: Human angle
    - renderer: Renderer instance
    - outputVideoPath: Path to save the output video
    - viewId: View ID
    - scanId: Scan ID
    - humanViewId: Human view ID
    - color: Color of the human mesh
    """
    writer = imageio.get_writer(outputVideoPath, fps=20)
    backgroundDepthPath = os.path.join(basicDataDir, "data/v1/scans", scanId, "matterport_skybox_images", f"{viewId}_skybox_depth_small.png")
    backgroundDepth = concat(backgroundDepthPath, DOWNSIZED_WIDTH)
    camLoc = (camLoc[0], camLoc[2], -camLoc[1])
    humanLoc = (humanLoc[0], humanLoc[2] - 1.36, -humanLoc[1])
    print(f"Camera location: {camLoc}, Camera angle: {camAngle}")
    print(f"Human location: {humanLoc}, Human angle: {humanAngle}")
    imgs = []

    thetaAngle = (np.pi / 180 * float(humanAngle))
    matrix = getRotation(theta=thetaAngle)
    for mesh in tqdm(meshes, desc=f"View_id {viewId}"):
        mesh.vertices = np.einsum("ij,ki->kj", matrix, mesh.vertices)
        mesh.vertices = mesh.vertices + humanLoc
        img = renderer.render(mesh, background, backgroundDepth, camLoc, camAngle, color=color)
        imgs.append(img)

    for cimg in imgs:
        writer.append_data(cimg)
    writer.close()

def renderFirstFrame(mesh, background, camLoc, camAngle, humanLoc, humanAngle, renderer, outputFramePath, viewId, scanId, humanViewId, color=[0, 0.8, 0.5]):
    """
    Render the first frame of a scene.
    
    Parameters:
    - mesh: Mesh to render
    - background: Background image
    - camLoc: Camera location
    - camAngle: Camera angle
    - humanLoc: Human location
    - humanAngle: Human angle
    - renderer: Renderer instance
    - outputFramePath: Path to save the output frame
    - viewId: View ID
    - scanId: Scan ID
    - humanViewId: Human view ID
    - color: Color of the human mesh
    """
    backgroundDepthPath = os.path.join(basicDataDir, "data/v1/scans", scanId, "matterport_skybox_images", f"{viewId}_skybox_depth_small.png")
    backgroundDepth = concat(backgroundDepthPath, DOWNSIZED_WIDTH)
    camLoc = (camLoc[0], camLoc[2], -camLoc[1])
    humanLoc = (humanLoc[0], humanLoc[2] - 1.36, -humanLoc[1])

    thetaAngle = (np.pi / 180 * float(humanAngle))
    matrix = getRotation(theta=thetaAngle)
    mesh.vertices = np.einsum("ij,ki->kj", matrix, mesh.vertices)
    mesh.vertices = mesh.vertices + humanLoc

    img = renderer.render(mesh, background, backgroundDepth, camLoc, camAngle, color=color)
    imageio.imwrite(outputFramePath, img)

def computeRel(srcLoc, tarLoc, currentHeading):
    """
    Compute the relative angle between the source and target locations.
    
    Parameters:
    - srcLoc: Source location
    - tarLoc: Target location
    - currentHeading: Current heading angle
    """
    targetHeading = (np.arctan2(tarLoc[0] - srcLoc[0], tarLoc[1] - srcLoc[1]) / (2 * np.math.pi)) * 360
    if targetHeading < 0:
        targetHeading += 360
    relAngle = currentHeading - targetHeading
    if abs(relAngle) > 180:
        if relAngle > 0:
            relAngle -= 360
        else:
            relAngle += 360
    relAngle /= 2
    return relAngle - currentHeading

def heFusion(inputPath, outputVideoPath, bgdImgPath, viewId, camLoc, humanLoc, camHeading, humanAngle, scanId, humanViewId):
    """
    Perform human-environment fusion and render a video.
    
    Parameters:
    - inputPath: Input path for the meshes
    - outputVideoPath: Output path for the video
    - bgdImgPath: Path to the background image
    - viewId: View ID
    - camLoc: Camera location
    - humanLoc: Human location
    - camHeading: Camera heading
    - humanAngle: Human angle
    - scanId: Scan ID
    - humanViewId: Human view ID
    """
    meshes = []
    objFiles = [f for f in os.listdir(inputPath) if f.endswith('.obj')]
    sortedObjFiles = sorted(objFiles)
    for objFile in sortedObjFiles[:60]:
        objPath = os.path.join(inputPath, objFile)
        mesh = trimesh.load(objPath)
        meshes.append(mesh)

    background = imageio.imread(bgdImgPath)
    camAngle = camHeading
    width = background.shape[1]
    height = background.shape[0]
    renderer = getRenderer(width, height)

    renderVideo(meshes, background, camLoc, camAngle, humanLoc, humanAngle, renderer, outputVideoPath, viewId, scanId, humanViewId)

def renderFrames(meshes, background, backgroundDepth, camLoc, camAngle, camElevation, humanLoc, humanAngle, renderer, viewId, color=[0, 0.8, 0.5]):
    """
    Render frames for the given meshes and background.
    
    Parameters:
    - meshes: List of meshes to render
    - background: Background image
    - backgroundDepth: Depth of the background
    - camLoc: Camera location
    - camAngle: Camera angle
    - camElevation: Camera elevation
    - humanLoc: Human location
    - humanAngle: Human angle
    - renderer: Renderer instance
    - viewId: View ID
    - color: Color of the human mesh
    """
    backgroundDepth *= 0.25 * 0.5
    camLoc = (camLoc[0], camLoc[2], -camLoc[1])
    humanLoc = (humanLoc[0], humanLoc[2] - 1.36, -humanLoc[1])
    print(f"Camera location: {camLoc}, Camera angle: {camAngle}, Camera elevation: {camElevation}")
    print(f"Human location: {humanLoc}, Human angle: {humanAngle}")
    imgs = []
    thetaAngle = (np.pi / 180 * float(humanAngle))
    matrix = getRotation(theta=thetaAngle)
    for mesh in meshes:
        mesh.vertices = np.einsum("ij,ki->kj", matrix, mesh.vertices)
        mesh.vertices = mesh.vertices + humanLoc
        img, _ = renderer.render_agent(mesh, background, backgroundDepth, camLoc, camAngle, camElevation, color=color)
        imgs.append(img)
    return imgs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path', default=None, help='Input file location')
    parser.add_argument('-o', '--output_path', default=None, help='Output file location')
    parser.add_argument('-bgi', '--background_image_path', default=None, help='Background image file location')
    opt = parser.parse_args()

if __name__ == "__main__":
    main()
