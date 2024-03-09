# HC-VLN_simulator

This repo is for add 3D human in real-world environment

## Create conda environment
```bash
conda create --name hc3d_simulater python=3.10
conda activate hc3d_simulater
pip install -r requirements.txt
```
## Download dataset
To use the simulator you must first download the [Matterport3D Dataset](https://niessner.github.io/Matterport/) which is available after requesting access [here](https://niessner.github.io/Matterport/). The download script that will be provided allows for downloading of selected data types. At minimum you must download the `matterport_skybox_images`. If you wish to use depth outputs then also download `undistorted_depth_images` and `undistorted_camera_parameters`.
download Matterport3D dataset from https://niessner.github.io/Matterport/
get download_mp.py
```bash
mkdir $HC3D_SIMULATOR_DTAT_PATH/MP3D_dataset
python2 download_mp.py -o $HC3D_SIMULATOR_DTAT_PATH/MP3D_dataset --type matterport_skybox_images undistorted_camera_parameters undistorted_depth_images
python unzip_data.py
```
## Set environment
Set an environment variable to the location of the **unzipped** dataset, where <PATH> is the full absolute path (not a relative path or symlink) to the directory containing the individual matterport scan directories (17DRP5sb8fy, 2t7WUuJeko7, etc):
```bash
export HC3D_SIMULATOR_DTAT_PATH=/your/path/to/store/data && echo $HC3D_SIMULATOR_DTAT_PATH
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

## HC3D Simulator GUI
```bash
python GUI.py
```
## Human motion generation
see human_motion_model/README
## human-scene fusion
see human-viewpoint_pair/README


Pyrender supports three backends for offscreen rendering:
  Pyglet, the same engine that runs the viewer. This requires an active display manager, so you can’t run it on a headless server.
  OSMesa, a software renderer.
  EGL, which allows for GPU-accelerated rendering without a display manager.
  default is EGL
pyrender offscreen rendering https://pyrender.readthedocs.io/en/latest/examples/offscreen.html

