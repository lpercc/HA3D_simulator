# HC-VLN_simulator

This repo is for add 3D human in real-world environment

## preparing dataset
download Matterport3D dataset from https://niessner.github.io/Matterport/
get download_mp.py
```
conda create -n mp3d python=2.7
conda activate mp3d
python download_mp.py -o ./Matterport3D_dataset --type matterport_skybox_images undistorted_camera_parameters
find ./Matterport3D_dataset/v1/scans -name '*.zip' -exec unzip -o '{}' -d ./data/v1/scans \;
conda deactivate
```


## How to run:
### 1. Create conda environment

```
conda create --name hcvln_simulater python=3.8
conda activate hcvln_simulater
```

### 2. install the following packages in your environnement:
```bash
pip install matplotlib
pip install torch
pip install ipdb
pip install sklearn
pip install pandas
pip install tqdm
pip install imageio
pip install pyyaml
pip install chumpy
pip install trimesh
pip install pyrender
pip install imageio-ffmpeg
pip install opencv-python
```
### 3.concat skybox
```
python concat_skybox.py
```

### 4.Human motion generation

### 5.human-environment fusion(demo)
```bash
python fusion.py --mode run_single
```
Pyrender supports three backends for offscreen rendering:
  Pyglet, the same engine that runs the viewer. This requires an active display manager, so you can’t run it on a headless server.
  OSMesa, a software renderer.
  EGL, which allows for GPU-accelerated rendering without a display manager.
  default is EGL
pyrender offscreen rendering https://pyrender.readthedocs.io/en/latest/examples/offscreen.html

### 6.Create simulator

