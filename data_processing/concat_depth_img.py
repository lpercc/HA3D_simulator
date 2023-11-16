import cv2
import numpy as np
import os
import re

data_dir = "data/v1/scans"
scan_id = "17DRP5sb8fy"
view_id = "0e92a69a50414253a23043758f111cec"
view_depth=np.load(os.path.join(data_dir, scan_id, "matterport_panorama_depth_images", "00.npy")) * 1.4
human_depth = np.load(os.path.join(".","human_depth.npy")) * 0.86
view_img_path = os.path.join(data_dir, scan_id, "matterport_panorama_images", "{}.jpg".format(view_id))
human_depth_img_path = os.path.join(".","human_depth.png")

view_img = cv2.imread(view_img_path)
human_img = np.ones((1024, 4096, 3))*200

print(np.min(view_depth), np.max(view_depth))
print(np.min(human_depth), np.max(human_depth))
mask = (human_depth <= view_depth) & (human_depth != 0)
print(mask.shape,np.sum(human_depth)/np.sum(human_depth != 0))
# 扩展掩码到三个通道，以匹配rgb和background的形状
mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)


output_img = np.where(mask_3d, human_img,view_img)
cv2.imwrite("./fusion_depth.jpg",output_img)
