import os
import zipfile
import shutil
import tqdm as tqdm

file_path = os.path.join(os.environ.get("MATTERPORT_DATA_DIR"), "data/v1/scans")
dir_name_list = os.listdir(file_path)
print('unzip start')
bar = tqdm(dir_name_list, desc="unzip")
for dir_name in bar:
	bar.set_description()
	if not os.path.exists(os.path.join(file_path, dir_name, "undistorted_camera_parameters")):
		zip_path_1 = os.path.join(file_path, dir_name, "undistorted_camera_parameters.zip")
		file=zipfile.ZipFile(zip_path_1)
		file.extractall(os.path.join(file_path, dir_name))
		file.close()
		shutil.move(os.path.join(file_path, dir_name, dir_name, "undistorted_camera_parameters"), os.path.join(file_path, dir_name))
		os.rmdir(os.path.join(file_path, dir_name, dir_name))
	if not os.path.exists(os.path.join(file_path, dir_name, "matterport_skybox_images")):
		zip_path_2 = os.path.join(file_path, dir_name, "matterport_skybox_images.zip")
		file=zipfile.ZipFile(zip_path_2)
		file.extractall(os.path.join(file_path, dir_name))
		file.close()
		shutil.move(os.path.join(file_path, dir_name, dir_name, "matterport_skybox_images"), os.path.join(file_path, dir_name))
		os.rmdir(os.path.join(file_path, dir_name, dir_name))
	if not os.path.exists(os.path.join(file_path, dir_name, "undistorted_depth_images")):
		zip_path_3 = os.path.join(file_path, dir_name, "undistorted_depth_images.zip")
		file=zipfile.ZipFile(zip_path_3)
		file.extractall(os.path.join(file_path, dir_name))
		file.close()
		shutil.move(os.path.join(file_path, dir_name, dir_name, "undistorted_depth_images"), os.path.join(file_path, dir_name))
		os.rmdir(os.path.join(file_path, dir_name, dir_name))
print('unzip over')
