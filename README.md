# HA3D Simulator

HA3D Simulator is for add 3D human in real-world environment
The development of the simulator is based on：Matterport3D Simulator API, MDM

## Set environment
Set an environment variable to the location of the **unzipped** dataset, where <PATH> is the full absolute path (not a relative path or symlink) to the directory containing the individual matterport scan directories (17DRP5sb8fy, 2t7WUuJeko7, etc):
```bash
vim ~/.bashrc
export HA3D_SIMULATOR_DTAT_PATH=/your/path/to/store/data
source ~/.bashrc
echo $HA3D_SIMULATOR_DTAT_PATH
```
/your/path/to/store/data
--data
--human_motion_meshes
## Create conda environment
```bash
conda create --name ha3d_simulator python=3.10
conda activate ha3d_simulator
pip install -r requirements.txt
```

## Download dataset
To use the simulator you must first download the [Matterport3D Dataset](https://niessner.github.io/Matterport/) which is available after requesting access [here](https://niessner.github.io/Matterport/). The download script that will be provided allows for downloading of selected data types. At minimum you must download the `matterport_skybox_images`. If you wish to use depth outputs then also download `undistorted_depth_images` and `undistorted_camera_parameters`.
download Matterport3D dataset from https://niessner.github.io/Matterport/
get download_mp.py
```bash
python2 download_mp.py -o $HA3D_SIMULATOR_DTAT_PATH/dataset --type matterport_skybox_images undistorted_camera_parameters undistorted_depth_images
python scripts/unzip_data.py
```

## Dataset Preprocessing

To make data loading faster and to reduce memory usage we preprocess the `matterport_skybox_images` by downscaling and combining all cube faces into a single image. While still inside the docker container, run the following script:
```
./scripts/downsize_skybox.py
```

This will take a while depending on the number of processes used (which is a setting in the script). 

After completion, the `matterport_skybox_images` subdirectories in the dataset will contain image files with filename format `<PANO_ID>_skybox_small.jpg`. By default images are downscaled by 50% and 20 processes are used.

Precompute matching depth skybox images by running this script:
```
./scripts/depth_to_skybox.py
```

## Build Matterport3D Simulator
see Matterport3DSimulator/README

## HA3D Simulator
create pipe
```bash
mkdir pipe
mkfifo ./pipe/my_S2R_pipe
mkfifo ./pipe/my_R2S_pipe
```
run renderer
```bash
python HA3DRender.py --pipeID 0
```
Open another terminal and run HA3D Simulator GUI
```bash
python GUI.py
```

## Human motion generation
see human_motion_model/README
## human-scene fusion
see human-viewpoint_annotation/README


Pyrender supports three backends for offscreen rendering:
  Pyglet, the same engine that runs the viewer. This requires an active display manager, so you can’t run it on a headless server.
  OSMesa, a software renderer.
  EGL, which allows for GPU-accelerated rendering without a display manager.
  default is EGL
pyrender offscreen rendering https://pyrender.readthedocs.io/en/latest/examples/offscreen.html

## train
```bash
python tasks/DT_miniGPT/train_GPT.py --experiment_id time --cuda 2 --reward_strategy 1 --epochs 15 --fusion_type simple --target_rtg 5 --mode train 
```