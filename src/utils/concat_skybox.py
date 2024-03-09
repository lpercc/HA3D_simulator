import imageio

def concat(image_path, width):
    downsize_image = imageio.imread(image_path)[:,width:-width,:]
    return downsize_image
    