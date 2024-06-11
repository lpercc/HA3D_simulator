# HA3D Simulator

HA3D Simulator integrates 3D human models into real-world environments. Built upon Matterport3D Simulator API and MDM, this simulator offers a robust platform for immersive 3D simulations.

## Table of Contents
- [HA3D Simulator](#ha3d-simulator)
  - [Table of Contents](#table-of-contents)
  - [🔧 Setup Environment](#-setup-environment)
  - [🐍 Create Conda Environment](#-create-conda-environment)
  - [📥 Download Dataset](#-download-dataset)
  - [🔄 Dataset Preprocessing](#-dataset-preprocessing)
  - [🏗️ Build Matterport3D Simulator](#️-build-matterport3d-simulator)
  - [🚀 Run HA3D Simulator](#-run-ha3d-simulator)
  - [🕺 Human Motion Generation](#-human-motion-generation)
  - [🌆 Human-Scene Fusion](#-human-scene-fusion)
  - [🖥️ Offscreen Rendering](#️-offscreen-rendering)
  - [📊 Training](#-training)

## 🔧 Setup Environment
First, set an environment variable to the location of the **unzipped** dataset. Replace `<PATH>` with the full absolute path to the directory containing the individual Matterport scan directories.

```bash
vim ~/.bashrc
export HA3D_SIMULATOR_DTAT_PATH=/your/path/to/store/data
source ~/.bashrc
echo $HA3D_SIMULATOR_DTAT_PATH
```

Expected directory structure:
```
/your/path/to/store/data
├── data
└── human_motion_meshes
```

## 🐍 Create Conda Environment
Set up a Conda environment for the simulator.

```bash
conda create --name ha3d_simulator python=3.10
conda activate ha3d_simulator
pip install -r requirements.txt
```

## 📥 Download Dataset
To use the simulator, download the [Matterport3D Dataset](https://niessner.github.io/Matterport/) (access required).

```bash
python2 download_mp.py -o $HA3D_SIMULATOR_DTAT_PATH/dataset --type matterport_skybox_images undistorted_camera_parameters undistorted_depth_images
python scripts/unzip_data.py
```

## 🔄 Dataset Preprocessing
Speed up data loading and reduce memory usage by preprocessing the `matterport_skybox_images`.

```bash
./scripts/downsize_skybox.py
./scripts/depth_to_skybox.py
```

This script downscales and combines all cube faces into a single image, resulting in filenames like `<PANO_ID>_skybox_small.jpg`.

## 🏗️ Build Matterport3D Simulator
Follow the instructions in the [Matterport3DSimulator/README](Matterport3DSimulator/README).

## 🚀 Run HA3D Simulator
1. Create pipes:
    ```bash
    mkdir pipe
    mkfifo ./pipe/my_S2R_pipe
    mkfifo ./pipe/my_R2S_pipe
    ```

2. Run the renderer:
    ```bash
    python HA3DRender.py --pipeID 0
    ```

3. In a new terminal, start the HA3D Simulator GUI:
    ```bash
    python GUI.py
    ```

## 🕺 Human Motion Generation
Refer to the [human_motion_model/README](human_motion_model/README) for detailed instructions.

## 🌆 Human-Scene Fusion
Refer to the [human-viewpoint_pair/README](human-viewpoint_pair/README) for detailed instructions.

## 🖥️ Offscreen Rendering
Pyrender supports three backends for offscreen rendering:
- **Pyglet**: Requires an active display manager.
- **OSMesa**: Software renderer.
- **EGL**: GPU-accelerated rendering without a display manager (default).

More details: [Pyrender Offscreen Rendering](https://pyrender.readthedocs.io/en/latest/examples/offscreen.html)

## 📊 Training
Train the model using the following command:

```bash
python tasks/DT_miniGPT/train_GPT.py --experiment_id time --cuda 2 --reward_strategy 1 --epochs 15 --fusion_type simple --target_rtg 5 --mode train
```

---

Feel free to contribute, report issues, or request features!

