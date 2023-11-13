import cv2
import json
import numpy as np

def visualize_opencv(scan_id):
    global selected_point, text_box_content

    with open('../con/pos_info/{}_pos_info.json'.format(scan_id), 'r') as f:
        pos_info = json.load(f)

    image_size = 900
    grid_size = 35  # 每个网格的像素大小

    # 创建一个白色背景图像
    image = np.ones((image_size + 50, image_size, 3), np.uint8) * 255

    # 绘制网格和坐标轴
    for i in range(0, image_size, grid_size):
        cv2.line(image, (i, 0), (i, image_size), (200, 200, 200), 1, lineType=cv2.LINE_AA)
        cv2.line(image, (0, i), (image_size, i), (200, 200, 200), 1, lineType=cv2.LINE_AA)

    # 绘制坐标轴和箭头
    cv2.arrowedLine(image, (image_size // 2, image_size), (image_size // 2, 0), (0, 0, 0), 2)
    cv2.arrowedLine(image, (0, image_size // 2), (image_size, image_size // 2), (0, 0, 0), 2)

    # 将视点坐标转换为图像上的点，并在图像上标记为红色点
    points = {}
    for key in pos_info:
        x, y = int(image_size / 2 + pos_info[key][0] * grid_size), int(image_size / 2 - pos_info[key][1] * grid_size)
        points[key] = (x, y, pos_info[key])
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)

    selected_point = None
    text_box_content = ""

    # 显示图像
    cv2.imshow('Viewpoints', image)
    cv2.setMouseCallback('Viewpoints', on_click, [points, image])

    while True:
        # 更新文本框内容
        update_text_box(image, text_box_content)
        cv2.imshow('Viewpoints', image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            if selected_point:
                # 将选中的点变回红色
                x, y = points[selected_point][:2]
                cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
                selected_point = None
                text_box_content = ""
                update_text_box(image, text_box_content)

    cv2.destroyAllWindows()

def update_text_box(image, text):
    # 清除文本框区域
    image[-50:-1, :] = 255
    # 添加文本
    cv2.putText(image, text, (10, image.shape[1]+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

def on_click(event, x, y, flags, param):
    global selected_point, text_box_content

    points, image = param

    if event == cv2.EVENT_LBUTTONDOWN:
        for key, (px, py, coords) in points.items():
            if abs(x - px) < 10 and abs(y - py) < 10:
                if selected_point:
                    # 将先前选中的点变回红色
                    ox, oy = points[selected_point][:2]
                    cv2.circle(image, (ox, oy), 5, (0, 0, 255), -1)
                
                coord_text = f"({coords[0]:.2f}, {coords[1]:.2f})"
                selected_point = key
                text_box_content = key + coord_text
                # 将当前点变为蓝色，并显示坐标
                cv2.circle(image, (px, py), 5, (255, 0, 0), -1)
                
                #cv2.putText(image, coord_text, (px + 10, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                update_text_box(image, text_box_content)
                break

# Example usage
# visualize_opencv('example_scan_id') # Uncomment and replace 'example_scan_id' with actual scan id to use


# Example usage
# visualize_opencv('example_scan_id') # Uncomment and replace 'example_scan_id' with

scan_id = "17DRP5sb8fy"
visualize_opencv(scan_id)