

import math
import trimesh
import trimesh.transformations as tf
import pyrender
import numpy as np
from pyrender.constants import RenderFlags
import os

os.environ['PYOPENGL_PLATFORM'] = 'egl'

class Renderer:
    def __init__(self, background=None, resolution=(224, 224), bg_color=[0, 0, 0, 0.5], orig_img=False, wireframe=False):
        width, height = resolution
        self.background = np.zeros((height, width, 3))
        self.resolution = resolution

        self.orig_img = orig_img
        self.wireframe = wireframe
        self.renderer = pyrender.OffscreenRenderer(
            viewport_width=self.resolution[0],
            viewport_height=self.resolution[1],
            point_size=0.5
        )

        # set the scene
        self.scene = pyrender.Scene(bg_color=bg_color, ambient_light=(0.4, 0.4, 0.4))
        # set light
        self.light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=4)

    def render(self, mesh, background, cam_loc, cam_angle, human_loc=None, human_angle=None, mesh_filename=None, color=[1.0, 1.0, 0.9]):

        #Rx = trimesh.transformations.rotation_matrix(math.radians(0), [1, 0, 0])
        #mesh.apply_transform(Rx)

        if mesh_filename is not None:
            mesh.export(mesh_filename)

        if human_angle:
            R = trimesh.transformations.rotation_matrix(math.radians(human_angle), axis)
            mesh.apply_transform(R)

        # 这四个值分别代表x方向的缩放、y方向的缩放、x方向的平移和y方向的平移
        # sx, sy, tx, ty = cam
        # 弱透视相机
        """ 
        camera = WeakPerspectiveCamera(
            scale=[sx, sy],
            translation=[tx, ty],
            #设置相机的远裁剪面。这意味着在z=1000的位置之后的所有物体都不会被渲染
            zfar=1000.
        ) """



        camera = pyrender.PerspectiveCamera(yfov=np.pi / 2.0)

        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.7,
            alphaMode='OPAQUE',
            baseColorFactor=(color[0], color[1], color[2], 1.0)
        )

        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        mesh_node = self.scene.add(mesh, 'mesh')

        # 将刚刚创建的相机添加到场景中，并设置其位置和方向为camera_pose
        angle = np.radians(cam_angle)  # 旋转
        rotation_matrix = tf.rotation_matrix(angle, [0, 1, 0])
        dx, dy, dz = cam_loc # 例如，平移沿x轴，沿y轴，沿z轴
        translation_matrix = np.array([
            [1, 0, 0, dx],
            [0, 1, 0, dy],
            [0, 0, 1, dz],
            [0, 0, 0, 1]
        ])
        camera_pose = translation_matrix @ rotation_matrix
        
        # 光源
        light_pose = np.eye(4)
        light_pose[:3, 3] = [dx+1, dy, dz]
        light_node1 = self.scene.add(self.light, pose=light_pose.copy())

        light_pose[:3, 3] = [dx, dy+1, dz]
        light_node2 = self.scene.add(self.light, pose=light_pose.copy())

        light_pose[:3, 3] = [dx-1, dy, dz]
        light_node3 = self.scene.add(self.light, pose=light_pose.copy())

        cam_node = self.scene.add(camera, pose=camera_pose)



        if self.wireframe:
            render_flags = RenderFlags.RGBA | RenderFlags.ALL_WIREFRAME
        else:
            render_flags = RenderFlags.RGBA

        rgb, _ = self.renderer.render(self.scene, flags=render_flags)
        valid_mask = (rgb[:, :, -1] > 0)[:, :, np.newaxis]
        #print(img.shape, valid_mask.shape, rgb.shape)
        output_img = rgb[:, :, :-1] * valid_mask + (1 - valid_mask) * background
        image = output_img.astype(np.uint8)

        self.scene.remove_node(mesh_node)
        self.scene.remove_node(light_node1)
        self.scene.remove_node(light_node2)
        self.scene.remove_node(light_node3)
        self.scene.remove_node(cam_node)

        return image


def get_renderer(width, height):
    renderer = Renderer(resolution=(width, height),
                        bg_color=[1, 1, 1, 0.5],
                        orig_img=False,
                        wireframe=False)
    return renderer
