import cv2
import numpy as np
import os
import re
from collections import defaultdict
import argparse
from tqdm import tqdm

DOWNSIZED_WIDTH = 512
DOWNSIZED_HEIGHT = 512

# Constants
SKYBOX_WIDTH = 1024
SKYBOX_HEIGHT = 1024

def main(args):
    skybox_image_dir = args.data_dir
    #output_dir = "data/v1/skybox"

    scan_list = os.listdir(skybox_image_dir)

    for scan in scan_list:
        scan_file = os.path.join(skybox_image_dir, scan)
        #print(scan)
        input_dir = os.path.join(scan_file, "matterport_skybox_images")
        viewpoint_list = os.listdir(input_dir)
        # 使用正则表达式解析文件名，以获取全景视图编号和帧编号
        pattern = re.compile(r'(.+)_skybox(\d+)_sami\.jpg')
        panoramas = defaultdict(list)
        for viewpoint_file in viewpoint_list:
            match = pattern.match(viewpoint_file)
            #print(viewpoint_file)
            if match:
                #print(match.group(1))
                panoramas[match.group(1)].append(viewpoint_file)
                #print(match.group(1))
        output_dir = os.path.join(scan_file, "matterport_panorama_images")
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        for key in tqdm(panoramas, desc=f"scan_id {scan}"):
            #print(key)
            panoramas[key].sort(key=lambda x: int(pattern.match(x).group(2)))
            all_img = []
            #print(key,len(panoramas[key]))
            for i in range(1,5):
                img_path = os.path.join(input_dir, panoramas[key][i])
                skybox = cv2.imread(img_path)
                all_img.append(cv2.resize(skybox,(DOWNSIZED_WIDTH,DOWNSIZED_HEIGHT),interpolation=cv2.INTER_AREA))

            panoramic_view_img = cv2.hconcat(all_img)

            output_path = os.path.join(output_dir, key+".jpg")
            cv2.imwrite(output_path, panoramic_view_img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--data_dir',default= "./",help='data file location')
    args = parser.parse_args()
    main(args)