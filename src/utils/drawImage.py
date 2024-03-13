import numpy as np

def drawCentering(image, size, color = (0,0,255)):
    import cv2
    center_x, center_y = image.shape[1] // 2, image.shape[0] // 2
    # 设置准心的颜色为白色 (B, G, R)
    # 设置准心的大小
    # 画水平线
    image = cv2.line(image, (center_x - size, center_y), (center_x + size, center_y), color, 2)
    # 画垂直线
    image = cv2.line(image, (center_x, center_y - size), (center_x, center_y + size), color, 2)
