import numpy as np
import imageio
import os
import argparse
from tqdm import tqdm
import trimesh
from .renderer import get_renderer
#import cv2


def get_rotation(theta=np.pi):
    import src.utils.rotation_conversions as geometry
    import torch
    axis = torch.tensor([0, 1, 0], dtype=torch.float)
    axisangle = theta*axis
    matrix = geometry.axis_angle_to_matrix(axisangle)
    return matrix.numpy()

def render_video(meshes, background, cam_loc, cam_angle, human_loc, human_angle, renderer, output_video_path, view_id,color=[0, 0.8, 0.5]):
    writer = imageio.get_writer(output_video_path, fps=30)
    # Matterport3D坐标-->pyrende坐标
    cam_loc = (cam_loc[0], cam_loc[2], -cam_loc[1])
    human_loc = (human_loc[0], human_loc[2]-1.36, -human_loc[1])
    print(f"camera location:{cam_loc}, camera angle:{cam_angle}")
    print(f"human location:{human_loc}, human angle:{human_angle}")
    # human旋转矩阵
    theta_angle = (np.pi / 180 * float(human_angle))
    matrix = get_rotation(theta=theta_angle)
    imgs = []
    for mesh in tqdm(meshes, desc=f"View_id {view_id}"):
        #human旋转
        mesh.vertices = np.einsum("ij,ki->kj", matrix, mesh.vertices)
        #human平移
        mesh.vertices = mesh.vertices + human_loc

        img = renderer.render(mesh, background, cam_loc, cam_angle, color=color)
        imgs.append(img)

    for cimg in imgs:
        writer.append_data(cimg)
    writer.close()

def compute_rel(src_loc, tar_loc, current_heading):
    # convert to rel to y axis
    target_heading = (np.arctan2(tar_loc[0] - src_loc[0], tar_loc[1] - src_loc[1]) / (2*np.math.pi)) * 360
    if target_heading < 0:
        target_heading = target_heading + 360
    #print(target_heading)
    rel_angle = current_heading - target_heading
    if abs(rel_angle) > 180:
        if rel_angle > 0:
            rel_angle = rel_angle - 360
        else:
            rel_angle = 360 + rel_angle
    rel_angle = rel_angle / 2
    #print(rel_angle, current_heading)
    return rel_angle - current_heading

def HE_fusion(input_path, output_video_path, bgd_img_path, view_id, cam_loc, human_loc, cam_heading, human_angle=0):
    meshes = []
    # 从.obj文件创建mesh
    # 获取目录下的所有.obj文件，并按照序号从大到小排序
    obj_files = [f for f in os.listdir(input_path) if f.endswith('.obj')]
    #print(obj_files[0].split('frame')[1].split('.obj')[0])
    sorted_obj_files = sorted(obj_files)
    for obj_file in sorted_obj_files[:60]:
        obj_file.split('.')
        obj_path = os.path.join(input_path,obj_file)
        mesh = trimesh.load(obj_path)
        meshes.append(mesh)

    background = imageio.imread(bgd_img_path)
    #cv2.imwrite("./background.jpg",background)
    #print(background.shape)

    cam_angle = compute_rel(cam_loc, human_loc, cam_heading)
        
    width = background.shape[1]
    height = background.shape[0]
    renderer = get_renderer(width, height)

    render_video(meshes, background, cam_loc, cam_angle, human_loc, human_angle, renderer, output_video_path, view_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input_path',default= None,help='input file location')
    parser.add_argument('-o','--output_path',default= None,help='output file location')
    parser.add_argument('-bgi','--background_image_path',default= None,help='background image file location')
    opt = parser.parse_args()


if __name__ == "__main__":
    main()
