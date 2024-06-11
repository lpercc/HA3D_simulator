import math
import trimesh
import trimesh.transformations as tf
import pyrender
import numpy as np
from pyrender.constants import RenderFlags
import os
import cv2
from tqdm import tqdm
import inspect

def printFileAndLineQuick():
    """
    Quickly print the current file name and line number.
    """
    line_no = inspect.stack()[1][2]
    file_name = __file__
    print(f"File: {file_name}, Line: {line_no}")

# Set environment variable for OpenGL
os.environ['PYOPENGL_PLATFORM'] = 'egl'

class Renderer:
    def __init__(self, background=None, resolution=(224, 224), bgColor=[0, 0, 0, 0.5], origImg=False, wireframe=False):
        width, height = resolution
        self.background = np.zeros((height, width, 3))
        self.resolution = resolution
        self.origImg = origImg
        self.wireframe = wireframe
        self.renderer = pyrender.OffscreenRenderer(
            viewport_width=self.resolution[0],
            viewport_height=self.resolution[1],
            point_size=0.5
        )
        self.aspectRatio = self.resolution[0] / self.resolution[1]
        # Set the scene
        self.scene = pyrender.Scene(bg_color=bgColor, ambient_light=(0.4, 0.4, 0.4))
        # Set light
        self.light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=4)
        self.humanList = []
        self.humanLocation = []
        self.camNode = None
        self.lightNode1 = None

    def render(self, mesh, background, backgroundDepth, camLoc, camAngle, humanAngle=None, meshFilename=None, color=[1.0, 1.0, 0.9]):
        """
        Render a scene with the given parameters.
        
        Parameters:
        - mesh: The mesh to be rendered
        - background: The background image
        - backgroundDepth: The depth of the background
        - camLoc: Camera location
        - camAngle: Camera angle
        - humanAngle: Angle of the human (optional)
        - meshFilename: Filename to save the mesh (optional)
        - color: Color of the mesh
        """
        if meshFilename is not None:
            mesh.export(meshFilename)

        if humanAngle:
            R = trimesh.transformations.rotation_matrix(math.radians(humanAngle), [0, 1, 0])
            mesh.apply_transform(R)

        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.7,
            alphaMode='OPAQUE',
            baseColorFactor=(color[0], color[1], color[2], 1.0)
        )

        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
        meshNode = self.scene.add(mesh, 'mesh')

        dx, dy, dz = camLoc
        translationMatrix = np.array([
            [1, 0, 0, dx],
            [0, 1, 0, dy],
            [0, 0, 1, dz],
            [0, 0, 0, 1]
        ])

        lightPose = np.eye(4)
        lightPose[:3, 3] = [dx + 1, dy, dz]
        lightNode1 = self.scene.add(self.light, pose=lightPose.copy())

        lightPose[:3, 3] = [dx, dy + 1, dz]
        lightNode2 = self.scene.add(self.light, pose=lightPose.copy())

        lightPose[:3, 3] = [dx - 1, dy, dz]
        lightNode3 = self.scene.add(self.light, pose=lightPose.copy())

        camera = pyrender.PerspectiveCamera(yfov=np.pi / 2.0, aspectRatio=1)
        
        if self.wireframe:
            renderFlags = RenderFlags.ALL_WIREFRAME

        imageAll = []
        imageDepthAll = []
        camsAngle = [180, 90, 0, 270]

        for i in range(4):
            angle = np.radians(camsAngle[i] - camAngle)
            rotationMatrix = tf.rotation_matrix(angle, [0, 1, 0])
            cameraPose = translationMatrix @ rotationMatrix
            camNode = self.scene.add(camera, pose=cameraPose)
            image, dImg = self.renderer.render(self.scene)
            imageAll.append(image)
            imageDepthAll.append(dImg)
            self.scene.remove_node(camNode)

        rgb = cv2.hconcat(imageAll)
        humanDepth = cv2.hconcat(imageDepthAll) * 4000
        mask = (humanDepth <= backgroundDepth) & (humanDepth != 0)
        mask3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        outputImg = np.where(mask3d, rgb, background)
        image = outputImg.astype(np.uint8)

        self.scene.remove_node(meshNode)
        self.scene.remove_node(lightNode1)
        self.scene.remove_node(lightNode2)
        self.scene.remove_node(lightNode3)

        return image

    def newHumans(self, humanList, color=[0.7, 0.9, 0]):
        """
        Load new human meshes into the scene.

        Parameters:
        - humanList: List of human meshes
        - color: Color of the human meshes
        """
        self.humanList = []
        self.humanLocation = []
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.5,
            alphaMode='OPAQUE',
            baseColorFactor=(color[0], color[1], color[2], 1.0)
        )
        bar = tqdm(humanList, desc="Loading New Human of Building")
        for human in bar:
            location = human['location']
            meshes = []
            humanLocation = []
            humanStartLoc = (location[0], location[2] - 1.36, -location[1])
            thetaAngle = (np.pi / 180 * float(human['heading']))
            matrix = getRotation(theta=thetaAngle)
            minDist = 1
            oIndex = 0
            for index, item in enumerate(human['meshes'][0].vertices):
                sumDist = (item[0]**2) + (item[1]**2) + (item[2]**2)
                if sumDist < minDist:
                    minDist = sumDist
                    oIndex = index
            for mesh in human['meshes']:
                mesh.vertices = np.einsum("ij,ki->kj", matrix, mesh.vertices)
                mesh.vertices = mesh.vertices + humanStartLoc
                meshLocation = mesh.vertices[oIndex]
                humanLocation.append((meshLocation[0], -meshLocation[2], meshLocation[1] + 1.36))
                mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
                meshes.append(mesh)
            self.humanList.append(meshes)
            self.humanLocation.append(humanLocation)

    def newAgent(self, vfov, location, heading, elevation):
        """
        Add a new agent to the scene.

        Parameters:
        - vfov: Vertical field of view
        - location: Agent's location
        - heading: Agent's heading
        - elevation: Agent's elevation
        """
        if self.camNode is not None:
            self.scene.remove_node(self.camNode)
            self.scene.remove_node(self.lightNode1)
        camLoc = (location[0], location[2], -location[1])
        dx, dy, dz = camLoc
        translationMatrix = np.array([
            [1, 0, 0, dx],
            [0, 1, 0, dy],
            [0, 0, 1, dz],
            [0, 0, 0, 1]
        ])
        lightPose = np.eye(4)
        lightPose[:3, 3] = [dx, dy, dz]
        self.lightNode1 = self.scene.add(self.light, pose=lightPose.copy())
        camera = pyrender.PerspectiveCamera(yfov=vfov, aspectRatio=self.aspectRatio)
        rotationMatrixHeading = tf.rotation_matrix(heading, [0, -1, 0])
        rotationMatrixElevation = tf.rotation_matrix(elevation, [1, 0, 0])
        cameraPose = translationMatrix @ (rotationMatrixHeading @ rotationMatrixElevation)
        self.camNode = self.scene.add(camera, pose=cameraPose)

    def moveAgent(self, vfov, location, heading, elevation):
        """
        Move the agent to a new position.

        Parameters:
        - vfov: Vertical field of view
        - location: New location
        - heading: New heading
        - elevation: New elevation
        """
        self.scene.remove_node(self.camNode)
        self.scene.remove_node(self.lightNode1)
        camLoc = (location[0], location[2], -location[1])
        dx, dy, dz = camLoc
        translationMatrix = np.array([
            [1, 0, 0, dx],
            [0, 1, 0, dy],
            [0, 0, 1, dz],
            [0, 0, 0, 1]
        ])
        lightPose = np.eye(4)
        lightPose[:3, 3] = [dx, dy, dz]
        self.lightNode1 = self.scene.add(self.light, pose=lightPose.copy())
        camera = pyrender.PerspectiveCamera(yfov=vfov, aspectRatio=self.aspectRatio)
        rotationMatrixHeading = tf.rotation_matrix(heading, [0, -1, 0])
        rotationMatrixElevation = tf.rotation_matrix(elevation, [1, 0, 0])
        cameraPose = translationMatrix @ (rotationMatrixHeading @ rotationMatrixElevation)
        self.camNode = self.scene.add(camera, pose=cameraPose)

    def renderAgent(self, frameNum, background, backgroundDepth):
        """
        Render the agent at the current frame.

        Parameters:
        - frameNum: Frame number
        - background: Background image
        - backgroundDepth: Background depth
        """
        meshNodeList = []
        for mesh in self.humanList:
            meshNode = self.scene.add(mesh[frameNum], 'mesh')
            meshNodeList.append(meshNode)
        if self.wireframe:
            renderFlags = RenderFlags.ALL_WIREFRAME
        image, dImg = self.renderer.render(self.scene)
        for meshNode in meshNodeList:
            self.scene.remove_node(meshNode)
        rgb = image
        humanDepth = dImg * 4000
        mask = (humanDepth <= backgroundDepth) & (humanDepth != 0)
        mask3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        outputImg = np.where(mask3d, rgb, background)
        image = outputImg.astype(np.uint8)
        return image, humanDepth

    def getHumanLocation(self, frameNum):
        """
        Get the location of humans at the current frame.

        Parameters:
        - frameNum: Frame number
        """
        humanLocation = []
        for i in self.humanLocation:
            humanLocation.append(i[frameNum])
        return humanLocation

def getRenderer(width, height):
    """
    Create a new renderer instance.

    Parameters:
    - width: Width of the renderer
    - height: Height of the renderer
    """
    renderer = Renderer(resolution=(width, height),
                        bgColor=[1, 1, 1, 0.5],
                        origImg=False,
                        wireframe=False)
    return renderer

def getRotation(theta=np.pi):
    """
    Get a rotation matrix for a given angle.

    Parameters:
    - theta: Rotation angle
    """
    import src.utils.rotation_conversions as geometry
    import torch
    axis = torch.tensor([0, 1, 0], dtype=torch.float)
    axisAngle = theta * axis
    matrix = geometry.axisAngleToMatrix(axisAngle)
    return matrix.numpy()
