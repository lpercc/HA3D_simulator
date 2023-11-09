import cv2
import numpy as np
import os
import re
from collections import defaultdict


skybox_image_dir = "data/v1/scans"
#output_dir = "data/v1/skybox"

scan_list = os.listdir(skybox_image_dir)

for scan in scan_list:
    scan_file = os.path.join(skybox_image_dir, scan)
    print(scan)
    input_dir = os.path.join(scan_file, "matterport_skybox_images")
    viewpoint_list = os.listdir(input_dir)
    # 使用正则表达式解析文件名，以获取全景视图编号和帧编号
    pattern = re.compile(r'(.+)_skybox(\d+)_sami\.jpg')
    panoramas = defaultdict(list)
    for viewpoint_file in viewpoint_list:
        match = pattern.match(viewpoint_file)
        #print(viewpoint_file)
        if match:
            panoramas[match.group(1)].append(viewpoint_file)
            #print(match.group(1))
    for key in panoramas:
        print(key)
        panoramas[key].sort(key=lambda x: int(pattern.match(x).group(2)))
        all_img = []
        for i in range(1,5):
            img_path = os.path.join(input_dir, panoramas[key][i])
            img = cv2.imread(img_path)
            all_img.append(img)

        panoramic_view_img = cv2.hconcat(all_img)
        output_dir = os.path.join(scan_file, "matterport_panorama_images")
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        output_path = os.path.join(output_dir, key+".jpg")
        cv2.imwrite(output_path, panoramic_view_img)