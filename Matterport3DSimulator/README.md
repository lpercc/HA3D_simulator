# Matterport3D Simulator
Our simulator uses the API of the Matterport3D simulator, github repo: https://github.com/peteanderson80/Matterport3DSimulator We migrated some of our code here and fixed some bugs that were mainly caused by related library versions, such as opencv.

## Simulator Data

Matterport3D Simulator is based on densely sampled 360-degree indoor RGB-D images from the [Matterport3D dataset](https://niessner.github.io/Matterport/). The dataset consists of 90 different indoor environments, including homes, offices, churches and hotels. Each environment contains full 360-degree RGB-D scans from between 8 and 349 viewpoints, spread on average 2.25m apart throughout the entire walkable floorplan of the scene.

## Install Simulator

### dependencies
```bash
sudo apt-get install -y wget doxygen curl libjsoncpp-dev libepoxy-dev libglm-dev libosmesa6 libosmesa6-dev libglew-dev python3-setuptools cmake libopencv-dev python3-opencv libegl1-mesa-dev
```

### building
Ensure that you are in a conda environment(python=3.10): 
```bash
conda activate ha3d_simulator
```
build the simulator:
```
mkdir build && cd build
cmake -DEGL_RENDERING=ON ..
make
cd ../
```
The main requirements are:
- Ubuntu >= 14.04
- Nvidia-driver with CUDA installed 
- C++ compiler with C++11 support
- [CMake](https://cmake.org/) >= 3.10
- [OpenCV](http://opencv.org/)
- [OpenGL](https://www.opengl.org/)
- [GLM](https://glm.g-truc.net/0.9.8/index.html)
- [Numpy](http://www.numpy.org/)

Optional dependences (depending on the cmake rendering options):
- [OSMesa](https://www.mesa3d.org/osmesa.html) for OSMesa backend support
- [epoxy](https://github.com/anholt/libepoxy) for EGL backend support

### Rendering Options (GPU, CPU, off-screen)
Note that there are three rendering options, which are selected using [cmake](https://cmake.org/) options during the build process (by varying line 3 in the build commands immediately above):
- GPU rendering using OpenGL (requires an X server): `cmake ..` (default)
- Off-screen GPU rendering using [EGL](https://www.khronos.org/egl/): `cmake -DEGL_RENDERING=ON ..`
- Off-screen CPU rendering using [OSMesa](https://www.mesa3d.org/osmesa.html): `cmake -DOSMESA_RENDERING=ON ..`

The recommended (fast) approach for training agents is using off-screen GPU rendering (EGL).

## Set environment
```bash
vim ~/.bashrc
export PYTHONPATH=/your/path/to/Matterport3DSimulator/build
source ~/.bashrc
echo $PYTHONPATH
```

## Test
```
python
import MatterSim
```

## Acknowledgements

We would like to thank Matterport for allowing the Matterport3D dataset to be used by the academic community. This project is supported by a Facebook ParlAI Research Award and by the [Australian Centre for Robotic Vision](https://www.roboticvision.org/).
