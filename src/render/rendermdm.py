import numpy as np
import imageio
import os
import argparse
from tqdm import tqdm
import trimesh
from .renderer import get_renderer
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import cv2
import json

basic_data_dir = os.getenv('VLN_DATA_DIR')

def get_rotation(theta=np.pi):
    import src.utils.rotation_conversions as geometry
    import torch
    axis = torch.tensor([0, 1, 0], dtype=torch.float)
    axisangle = theta*axis
    matrix = geometry.axis_angle_to_matrix(axisangle)
    return matrix.numpy()

def adjust_cam_angle(image, cam_angle, human_angle):
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 将图片从BGR转换为RGB

    # 设置图像的显示大小（英寸）
    figsize = (16, 4)  # 例如，10英寸宽，8英寸高

    # 创建一个图表和坐标轴，并设置大小
    fig, ax = plt.subplots(figsize=figsize)

    # 显示图片
    ax.imshow(image)

    # 设置刻度
    ax.set_xticks(range(0, image.shape[1], int(image.shape[1] / 36)))  # 假设每10%宽度设置一个刻度
    ax.set_xticklabels(range(360, -1, -10))  # 假设刻度从0到360

    # 隐藏y轴刻度
    ax.get_yaxis().set_visible(False)

    # 显示图像
    plt.show()

    is_adjust = input("Is adjust(y/n)?")
    if is_adjust == 'y':
        add_angle = float(input("add_agent_angle:"))
        add_human_angle = float(input("add_human_angle(+):"))
        first_flag = True
    elif is_adjust == 'n':
        add_angle = 0
        add_human_angle = 0
        first_flag = False
    new_angle = add_angle+cam_angle
    new_human_angle = add_human_angle + human_angle
    return first_flag, new_angle, new_human_angle

def render_video(meshes, background, cam_loc, cam_angle, human_loc, human_angle, renderer, output_video_path, view_id,scan_id,human_view_id,color=[0, 0.8, 0.5]):
    writer = imageio.get_writer(output_video_path, fps=20)
    #0.25mm per unit
    background_depth = cv2.imread(os.path.join(basic_data_dir, "data/v1/scans", scan_id, "matterport_panorama_depth", f"{view_id}.png"), cv2.IMREAD_GRAYSCALE)
    # convert M
    background_depth = background_depth * 0.25 * 0.2
    #print(np.min(background_depth), np.max(background_depth))
    # Matterport3D坐标-->pyrende坐标
    cam_loc = (cam_loc[0], cam_loc[2], -cam_loc[1])
    human_loc = (human_loc[0], human_loc[2]-1.36, -human_loc[1])
    print(f"camera location:{cam_loc}, camera angle:{cam_angle}")
    print(f"human location:{human_loc}, human angle:{human_angle}")
    imgs = []

    first_flag = True

    theta_angle = (np.pi / 180 * float(human_angle))
    matrix = get_rotation(theta=theta_angle)
    for mesh in tqdm(meshes, desc=f"View_id {view_id}"):
        #human旋转
        mesh.vertices = np.einsum("ij,ki->kj", matrix, mesh.vertices)
        #human平移
        mesh.vertices = mesh.vertices + human_loc

        img = renderer.render(mesh, background, background_depth, cam_loc, cam_angle, color=color)
        imgs.append(img)

    for cimg in imgs:
        writer.append_data(cimg)
    writer.close()

def render_first_frame(mesh, background, cam_loc, cam_angle, human_loc, human_angle, renderer, output_frame_path, view_id,scan_id,human_view_id,color=[0, 0.8, 0.5]):
    #0.25mm per unit
    background_depth = cv2.imread(os.path.join(basic_data_dir, "data/v1/scans", scan_id, "matterport_panorama_depth", f"{view_id}.png"), cv2.IMREAD_GRAYSCALE)
    # convert M
    background_depth = background_depth * 0.25 * 0.2
    # Matterport3D坐标-->pyrende坐标
    cam_loc = (cam_loc[0], cam_loc[2], -cam_loc[1])
    human_loc = (human_loc[0], human_loc[2]-1.36, -human_loc[1])
    #print(f"camera location:{cam_loc}, camera angle:{cam_angle}")
    #print(f"human location:{human_loc}, human angle:{human_angle}")
    # 每个建筑场景中的视点视角朝向

    theta_angle = (np.pi / 180 * float(human_angle))
    matrix = get_rotation(theta=theta_angle)
    #human旋转
    mesh.vertices = np.einsum("ij,ki->kj", matrix, mesh.vertices)
    #human平移
    mesh.vertices = mesh.vertices + human_loc

    img = renderer.render(mesh, background, background_depth, cam_loc, cam_angle, color=color)
    
    imageio.imwrite(output_frame_path, img)

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

def HE_fusion(input_path, output_video_path, bgd_img_path, view_id, cam_loc, human_loc, cam_heading, human_angle,scan_id,human_view_id):
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

    #cam_angle = compute_rel(cam_loc, human_loc, cam_heading)
    cam_angle = cam_heading
    width = background.shape[1]
    height = background.shape[0]
    renderer = get_renderer(width, height)

    render_video(meshes, background, cam_loc, cam_angle, human_loc, human_angle, renderer, output_video_path, view_id, scan_id,human_view_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input_path',default= None,help='input file location')
    parser.add_argument('-o','--output_path',default= None,help='output file location')
    parser.add_argument('-bgi','--background_image_path',default= None,help='background image file location')
    opt = parser.parse_args()


if __name__ == "__main__":
    main()
