import cv2
import numpy as np
import os
import re
from collections import defaultdict

DOWNSIZED_WIDTH = 512
DOWNSIZED_HEIGHT = 512

# Constants
SKYBOX_WIDTH = 1024
SKYBOX_HEIGHT = 1024

input_dir = "data/v1/scans"
#output_dir = "data/v1/skybox"

scan_list = os.listdir(input_dir)

for scan in scan_list:
    scan_file = os.path.join(input_dir, scan)
    print(scan)
    skybox_depth_dir = os.path.join(scan_file, "matterport_panorama_depth")
    viewpoint_list = os.listdir(skybox_depth_dir)
    for viewpoint_file in viewpoint_list:
        depth_path = os.path.join(skybox_depth_dir, viewpoint_file)
        depth_img = cv2.imread(depth_path)[:,DOWNSIZED_WIDTH:-DOWNSIZED_WIDTH,:]
        print(depth_img.shape)
        output_dir = os.path.join(scan_file, "matterport_panorama_depth")
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        view_id = viewpoint_file.split("_")[0]
        output_path = os.path.join(output_dir, view_id+".png")
        print(view_id)
        cv2.imwrite(output_path, depth_img)

