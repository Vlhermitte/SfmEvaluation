# Structure-from-Motion (SfM) Evaluation Protocol

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Evaluation Protocol](#evaluation-protocol)
    - [Camera Pose Error](#camera-pose-error)
    - [Novel View Synthesis (in progress)](#novel-view-synthesis-in-progress)
- [Input Formats](#input-formats)
- [Output Formats](#output-formats)

## Overview
This project provides tools for reading, writing, and evaluating Structure-from-Motion (SfM) models.
The evaluation protocol is composed of the following elements:
* Camera pose error
* Novel view synthesis (in progress)

## Installation
In order to use the project, you need to execute the following script:
```
bash scripts/get_datasets.sh
```
This script will download the ETH3D dataset.

```
bash scripts/clone_repos.sh
```
This script will clone the following repositories:
* GLOMAP
* VGGSfm
* Flowmap
* Ace0

Once you have cloned the repositories, you need to install the dependencies for each method.
Follow the instructions in the README files of each repository. Most of the methods only require to create a virtual environment and install the dependencies (conda or pip).

GLOMAP however requires to compile the code. Or to download the precompiled binaries (for Windows only).

## Usage
To run the evaluation on all results, run the following command:
```
bash scripts/evaluate.sh
```
The script expect the result to be stored in data/results/\<methods>/ETH3D/\<scene> directory.
For more details, see the script [`evaluate.sh`](scripts/evaluate.sh) in the [`scripts`](scripts) directory.

To run an individual evaluation, the [`main.py`](src/main.py) file can be used.
```
python main.py --gt-model-path <path_to_gt_model> --est-model-path <path_to_estimated_model>
```

An absolute pose evaluation script is **under development** (see [`absolute_error_evaluation.py`](src/evaluation/absolute_error_evaluation.py)).
It first tries to perform alignment between the estimated and ground truth camera poses using the Kabsch-Umeyama algorithm.
Then, it computes the absolute rotation and translation errors.


## Evaluation Protocol

### Camera Pose Error
The evaluation protocol assesses the camera relative rotation error (RRE) and relative translation error (RTE) 
as the computed angle between the ground truth and estimated camera translation vector. 
Specifically, for each pair of images \(i\) and \(j\), the relative rotation and translation are computed as follows:
```math
  R_{rel} = R_j \cdot R_i^T
```
```math
  t_{rel} = t_j - (R_{rel} \cdot t_i)
```
For more details, see the function [`evaluate_relative_errors`](src/evaluation/relative_error_evaluation.py) in [`evaluation.py`](src/evaluation/relative_error_evaluation.py).

### Novel View Synthesis (in progress)
The evaluation protocol assesses the quality of the novel view synthesis by comparing rendered images with the ground truth images.
In order to produce novel view synthesis, nerfstudio is used to render the scene using NeRF or Gaussian Splatting.
The evaluation is performed by computing the PSNR and SSIM between the rendered image and the ground truth image.

**_For now this part is under development and the evaluation is not yet implemented._**
#### Usage
We provide a python file to run the training and evaluation [`run_nerfstudio.py`](src/run_nerfstudio.py):
```
python run_nerfstudio.py --dataset-path PATH_TO_SCENE_IMAGES --results-path PATH_TO_RESULTS
```
The script expect the PATH_TO_SCENE_IMAGES to be the path to the dataset with the images directory, 
and the PATH_TO_RESULTS to be the path to the results of the SFM method (e.g. data/results/glomap/ETH3D/courtyard/colmap/sparse/0).

The files in PATH_TO_RESULTS are expected to be in **COLMAP** format (.txt/.bin files).

## Input Formats
The project takes as input a COLMAP model and outputs the evaluation results.

## Output Formats
* GLOMAP output as a COLMAP model
* VGGSfm output as a COLMAP model
* Flowmap output as a COLMAP model (see python file overfit.py line 120)
* Ace0 output as a .txt file
  * Camera poses are world-to-camera transformations using OpenCV convention

### COLMAP Model

The COLMAP model are a .txt/.bin files that contains the following information:
* Camera Pose (rotation as quaternion and translation) in world coordinates
* 3D points in world coordinates

## Dataset used
- ETH3D Stereo dataset (https://www.eth3d.net/datasets)
  - Info available are: (in meters)
    - Ground truth camera poses
    - Camera intrinsics
    - Ground truth 3D points
    - COLMAP model
- Tanks and Temples dataset (https://www.tanksandtemples.org/download/)
  - Info available are:
    - Ground truth 3D points
    - .ply file
- MipNerf dataset (https://jonbarron.info/mipnerf360/)
  - Info available are:
    - COLMAP model

### Notes
- In the Tanks and Temples dataset, if the scene is an object (e.g. Ignatius, Truck), the ground truth 3D points \
    is a point cloud of the object without the background.

## TODO
- [x] Implement relative rotation and translation error evaluation
- [ ] Implement 3D triangulation error evaluation (accuracy and completeness of 3D triangulation)
- [ ] Implement novel view synthesis evaluation


# Resources
- [COLMAP](https://colmap.github.io/)
- [ETH3D](https://www.eth3d.net/)
- [Tanks and Temples](https://www.tanksandtemples.org/)
- [MipNerf](https://jonbarron.info/mipnerf360/)
- [TUM RGB-D SLAM tools](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/tools/)
- [Novel View Synthesis](https://arxiv.org/abs/1601.06950)