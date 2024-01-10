import cv2
import os
import argparse
from tqdm import tqdm

DOWNSIZED_WIDTH = 512
DOWNSIZED_HEIGHT = 512

# Constants
SKYBOX_WIDTH = 1024
SKYBOX_HEIGHT = 1024

def main(args):
    input_dir = os.path.join(args.data_dir, "data/v1/scans")
    #output_dir = "data/v1/skybox"

    scan_list = os.listdir(input_dir)

    for scan in scan_list:
        scan_file = os.path.join(input_dir, scan)
        skybox_depth_dir = os.path.join(scan_file, "matterport_panorama_images")
        viewpoint_list = os.listdir(skybox_depth_dir)
        output_dir = os.path.join(scan_file, "matterport_panorama_depth")
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        for viewpoint_file in tqdm(viewpoint_list, desc=f"scan_id {scan}"):
            if viewpoint_file.endswith(".png"):
                depth_path = os.path.join(skybox_depth_dir, viewpoint_file)
                depth_img = cv2.imread(depth_path)[:,DOWNSIZED_WIDTH:-DOWNSIZED_WIDTH,:]
                if os.path.exists(depth_path):
                    os.remove(depth_path)
                view_id = viewpoint_file.split("_")[0]
                output_path = os.path.join(output_dir, view_id+".png")
                cv2.imwrite(output_path, depth_img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--data_dir',default= "./",help='data file location')
    args = parser.parse_args()
    main(args)