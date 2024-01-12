import os

def rename_folders(directory):
    # 遍历目录中的所有项
    for name in os.listdir(directory):
        # 拼接完整的文件/文件夹路径
        old_path = os.path.join(directory, name)

        # 确保它是一个文件夹
        if os.path.isdir(old_path):
            # 生成新的文件夹名，将空格替换为下划线
            new_name = name.replace(' ', '_')
            new_path = os.path.join(directory, new_name)

            # 重命名文件夹
            os.rename(old_path, new_path)
            print(f"Renamed '{old_path}' to '{new_path}'")

# 替换为您的目录路径
basic_data_dir = "/media/lmh/backend/HC-VLN_dataset"
motion_model_dir = os.path.join(basic_data_dir,"human_motion_meshes")
rename_folders(motion_model_dir)