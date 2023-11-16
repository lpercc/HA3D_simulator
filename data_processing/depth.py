import numpy as np
import os
from PIL import Image
import cv2

target = 'sample_pc/0000_depth.npy'
rgb_filename = 'sample_pc/0000_depth.png'
u = 100
v = 10

depth = np.load(target).T
depth = np.flip(depth, axis=0)

depth_img = cv2.imread(rgb_filename)
#depth_img[u,v] = [255,91,255]
imGray = cv2.cvtColor(depth_img, cv2.COLOR_BGR2GRAY)

cv2.imwrite("./sample_pc/gray.jpg",imGray)

print(depth.shape, depth_img.shape, imGray.shape)
print(depth[u,v], depth_img[u,v], imGray[u,v])