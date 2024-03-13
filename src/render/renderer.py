

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

def print_file_and_line_quick():
    # 快速获取当前行号
    line_no = inspect.stack()[1][2]
    # 快速获取当前文件名
    file_name = __file__
    print(f"File: {file_name}, Line: {line_no}")

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
        self.human_list = []
        self.cam_node = None
        self.light_node1 = None
    def render(self, mesh, background, background_depth, cam_loc, cam_angle, human_angle=None, mesh_filename=None, color=[1.0, 1.0, 0.9]):

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



        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.7,
            alphaMode='OPAQUE',
            baseColorFactor=(color[0], color[1], color[2], 1.0)
        )

        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        mesh_node = self.scene.add(mesh, 'mesh')
        

        dx, dy, dz = cam_loc # 例如，平移沿x轴，沿y轴，沿z轴
        translation_matrix = np.array([
            [1, 0, 0, dx],
            [0, 1, 0, dy],
            [0, 0, 1, dz],
            [0, 0, 0, 1]
        ])
        
        
        # 光源
        light_pose = np.eye(4)
        light_pose[:3, 3] = [dx+1, dy, dz]
        light_node1 = self.scene.add(self.light, pose=light_pose.copy())

        light_pose[:3, 3] = [dx, dy+1, dz]
        light_node2 = self.scene.add(self.light, pose=light_pose.copy())

        light_pose[:3, 3] = [dx-1, dy, dz]
        light_node3 = self.scene.add(self.light, pose=light_pose.copy())

        camera = pyrender.PerspectiveCamera(yfov=np.pi / 2.0, aspectRatio=1)
        
        if self.wireframe:
            render_flags = RenderFlags.ALL_WIREFRAME


        #cam_nodes = []
        image_all = []
        image_depth_all = []
        cams_angle = [180,90,0,270]
        for i in range(4):
            angle = np.radians(cams_angle[i]-cam_angle)  # 旋转
            rotation_matrix = tf.rotation_matrix(angle, [0, 1, 0])
            camera_pose = translation_matrix @ rotation_matrix
            # 将刚刚创建的相机添加到场景中，并设置其位置和方向为camera_pose
            cam_node = self.scene.add(camera, pose=camera_pose)
            #cam_nodes.append(cam_node)
            image, d_img = self.renderer.render(self.scene)
            image_all.append(image)
            image_depth_all.append(d_img)
            #print(image.shape)
            self.scene.remove_node(cam_node)


        #rgb = cv2.hconcat([image_all[2], image_all[3], image_all[0], image_all[1]])
        rgb = cv2.hconcat(image_all)
        human_depth = cv2.hconcat(image_depth_all) * 0.8
        #print(np.max(human_depth))
        #print(human_depth.shape, background_depth.shape)
        mask = (human_depth <= background_depth) & (human_depth != 0)
        #print(mask.shape,np.sum(human_depth)/np.sum(human_depth != 0))
        # 扩展掩码到三个通道，以匹配rgb和background的形状
        mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

        # 创建输出图像，其中白色背景被 background 替换
        # 注意这里只处理RGB通道，不处理alpha通道
        output_img = np.where(mask_3d, rgb, background)
        #cv2.imwrite("./background.jpg",background)
        #cv2.imwrite("./rgb_valid_mask.jpg",rgb[:, :, :-1] * valid_mask)
        #cv2.imwrite("./background_valid_mask.jpg",(1 - valid_mask) * background)
        #cv2.imwrite("./output_img.jpg",output_img)
        image = output_img.astype(np.uint8)

        self.scene.remove_node(mesh_node)
        self.scene.remove_node(light_node1)
        self.scene.remove_node(light_node2)
        self.scene.remove_node(light_node3)
        
        #np.save("./depth.npy",depth)

        return image
    
    def newHumans(self, human_list, color=[1, 0.2, 0]):
        self.human_list = []
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.7,
            alphaMode='OPAQUE',
            baseColorFactor=(color[0], color[1], color[2], 1.0)
        )
        bar = tqdm(human_list, desc=f"Loading New Human of Building")
        for human in bar:
            location = human['location']
            meshes = []
            human_loc = (location[0], location[2]-1.36, -location[1])
            theta_angle = (np.pi / 180 * float(human['heading']))
            matrix = get_rotation(theta=theta_angle)
            for mesh in human['meshes']:
                mesh.vertices = np.einsum("ij,ki->kj", matrix, mesh.vertices)
                #human平移
                mesh.vertices = mesh.vertices + human_loc
                mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
                meshes.append(mesh)
            self.human_list.append(meshes)

    def newAgent(self, vfov, location, heading, elevation):
        if  self.cam_node is not None:
            self.scene.remove_node(self.cam_node)
            self.scene.remove_node(self.light_node1)
        cam_loc = (location[0], location[2], -location[1])
        dx, dy, dz = cam_loc # 例如，平移沿x轴，沿y轴，沿z轴
        translation_matrix = np.array([
            [1, 0, 0, dx],
            [0, 1, 0, dy],
            [0, 0, 1, dz],
            [0, 0, 0, 1]
        ])
        # 光源
        light_pose = np.eye(4)
        light_pose[:3, 3] = [dx, dy, dz]
        self.light_node1 = self.scene.add(self.light, pose=light_pose.copy())
        camera = pyrender.PerspectiveCamera(yfov=vfov, aspectRatio=1)
        #cam_nodes = []
        rotation_matrix_heading = tf.rotation_matrix(heading, [0, -1, 0])
        rotation_matrix_elevation = tf.rotation_matrix(elevation, [1, 0, 0])
        camera_pose = translation_matrix @ (rotation_matrix_heading @ rotation_matrix_elevation)
        # 将刚刚创建的相机添加到场景中，并设置其位置和方向为camera_pose
        self.cam_node = self.scene.add(camera, pose=camera_pose)

    def moveAgent(self, vfov, location, heading, elevation):
        self.scene.remove_node(self.cam_node)
        self.scene.remove_node(self.light_node1)
        cam_loc = (location[0], location[2], -location[1])
        dx, dy, dz = cam_loc # 例如，平移沿x轴，沿y轴，沿z轴
        translation_matrix = np.array([
            [1, 0, 0, dx],
            [0, 1, 0, dy],
            [0, 0, 1, dz],
            [0, 0, 0, 1]
        ])
        # 光源
        light_pose = np.eye(4)
        light_pose[:3, 3] = [dx, dy, dz]
        self.light_node1 = self.scene.add(self.light, pose=light_pose.copy())
        camera = pyrender.PerspectiveCamera(yfov=vfov, aspectRatio=1)
        #cam_nodes = []
        rotation_matrix_heading = tf.rotation_matrix(heading, [0, -1, 0])
        rotation_matrix_elevation = tf.rotation_matrix(elevation, [1, 0, 0])
        camera_pose = translation_matrix @ (rotation_matrix_heading @ rotation_matrix_elevation)
        # 将刚刚创建的相机添加到场景中，并设置其位置和方向为camera_pose
        self.cam_node = self.scene.add(camera, pose=camera_pose)

    def render_agent(self, frame_num, background, background_depth):
        mesh_node_list = []
        for mesh in self.human_list:
            mesh_node = self.scene.add(mesh[frame_num], 'mesh')
            mesh_node_list.append(mesh_node)
        if self.wireframe:
            render_flags = RenderFlags.ALL_WIREFRAME
        image, d_img = self.renderer.render(self.scene)
        #print(image.shape)
        for mesh_node in mesh_node_list:
            self.scene.remove_node(mesh_node)
        rgb = image
        human_depth = d_img
        mask = (human_depth <= background_depth) & (human_depth != 0)
        # 扩展掩码到三个通道，以匹配rgb和background的形状
        mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

        # 创建输出图像，其中白色背景被 background 替换
        # 注意这里只处理RGB通道，不处理alpha通道
        output_img = np.where(mask_3d, rgb, background)
        image = output_img.astype(np.uint8)
        return image,human_depth



def get_renderer(width, height):
    renderer = Renderer(resolution=(width, height),
                        bg_color=[1, 1, 1, 0.5],
                        orig_img=False,
                        wireframe=False)
    return renderer

def get_rotation(theta=np.pi):
    import src.utils.rotation_conversions as geometry
    import torch
    axis = torch.tensor([0, 1, 0], dtype=torch.float)
    axisangle = theta*axis
    matrix = geometry.axis_angle_to_matrix(axisangle)
    return matrix.numpy()