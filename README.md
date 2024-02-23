# HC-VLN_simulator

This repo is for add 3D human in real-world environment
clone this repo
```bash
cd HC-VLN_simulator
```

## Preparing dataset
download Matterport3D dataset from https://niessner.github.io/Matterport/
get download_mp.py
```bash
conda create -n mp3d python=2.7
conda activate mp3d
mkdir ./Matterport3D_dataset
python ./HC-VLN_simulator/download_mp.py -o ./Matterport3D_dataset --type matterport_skybox_images undistorted_camera_parameters  undistorted_depth_images
find ./Matterport3D_dataset/v1/scans -name '*.zip' -exec unzip -o '{}' -d ./data/v1/scans \;
conda deactivate
```


## 1. Create conda environment

```bash
conda create --name hcvln_simulater python=3.8
conda activate hcvln_simulater
```

## 2. install the following packages in your environnement:
```bash
#pip install matplotlib
#pip install torch
#pip install ipdb
#pip install sklearn
#pip install pandas
pip install tqdm
pip install imageio
#pip install pyyaml
#pip install chumpy
pip install trimesh
pip install pyrender
#pip install imageio-ffmpeg
pip install opencv-python==4.1.2.30
#pip install PyQt5 PyQtChart
```
## 3.concat skybox
```bash
python concat_skybox.py
```

## 4.Human motion generation
### collacte human descriptions
```human_motion_text.json```
### generate human motion backbone
get MDM repo from https://github.com/GuyTevet/motion-diffusion-model
```bash
cd ..
git clone https://github.com/GuyTevet/motion-diffusion-model.git
cd motion-diffusion-model
```
follow the README, after step 3 (https://github.com/GuyTevet/motion-diffusion-model#3-download-the-pretrained-models)
run the script
```bash
#copy prompts file
cp ../HC-VLN_simulator/HC-VLN_text_prompts.txt ./assets/
#generate from human motion text prompts
python -m sample.generate --model_path ./save/humanml_trans_enc_512/model000200000.pt --input_text ./assets/HC-VLN_text_prompts.txt --num_repetitions 3 --batch_size 145
```
You may also define:
  --device id.
  --seed to sample different prompts.
  --motion_length (text-to-motion only) in seconds (maximum is 9.8[sec]).

### screen human motion backbone(manual)

### Render SMPL mesh
```bash
cp ../HC-VLN_simulator/backbone2smpl.py ./visualize
cp ../HC-VLN_simulator/human_motion_text.json ./
python -m visualize.backbone2smpl
```

## 5.human-environment fusion(demo)
```bash
python fusion.py --mode run_single
```
## 5.1 GUI
```bash
python GUI.py
```

Pyrender supports three backends for offscreen rendering:
  Pyglet, the same engine that runs the viewer. This requires an active display manager, so you can’t run it on a headless server.
  OSMesa, a software renderer.
  EGL, which allows for GPU-accelerated rendering without a display manager.
  default is EGL
pyrender offscreen rendering https://pyrender.readthedocs.io/en/latest/examples/offscreen.html

## 6.Create simulator

