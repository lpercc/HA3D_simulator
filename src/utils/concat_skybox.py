import imageio

def concat(image_path, width):
    image = imageio.imread(image_path)
    if len(image.shape) == 3:  # 检查是否为三通道图像
        downsize_image = image[:, width:-width, :]
    elif len(image.shape) == 2:  # 检查是否为二通道图像
        downsize_image = image[:, width:-width]
    else:
        raise ValueError("Unsupported image dimensions")
    return downsize_image

def concat_feet(image_path, width):
    image = imageio.imread(image_path)
    if len(image.shape) == 3:  # 检查是否为三通道图像
        downsize_image = image[:, -width:-1, :]
    elif len(image.shape) == 2:  # 检查是否为二通道图像
        downsize_image = image[:, -width:-1]
    else:
        raise ValueError("Unsupported image dimensions")
    return downsize_image
