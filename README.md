# Structure-from-Motion (SfM) Evaluation Protocol

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Evaluation Protocol](#evaluation-protocol)
    - [Camera Pose Error](#camera-pose-error)
    - [Novel View Synthesis (In Progress)](#novel-view-synthesis-in-progress)

## Overview
This project provides tools for reading, writing, and evaluating Structure-from-Motion (SfM) models.
The evaluation protocol consists of the following components:
- **Camera Pose Error**
- **Novel View Synthesis** *(in progress)*

## Installation
To set up the project, execute the following scripts:

### Step 1: Download the ETH3D Dataset
```bash
bash scripts/get_datasets.sh
```
This script will download the **ETH3D dataset** required for evaluation.

### Step 2: Clone Required Repositories
```bash
bash scripts/clone_repos.sh
```
This script will clone the following repositories:
- **GLOMAP**
- **VGGSfm**
- **Flowmap**
- **Ace0**

### Step 3: Install Dependencies
Each cloned repository contains a README file with installation instructions.
Follow these steps for each method:
- Most methods require creating a virtual environment and installing dependencies using **conda** or **pip**.
- **GLOMAP** requires compilation or downloading precompiled binaries (Windows only).

## Usage
### Running Evaluation on All Results
To evaluate all results, run:
```bash
bash scripts/evaluate.sh
```
The script expects results stored in:
```
data/results/<method>/ETH3D/<scene>/colmap/sparse/0
```
For details, see [`evaluate.sh`](scripts/evaluate.sh) in the [`scripts`](scripts) directory.

### Running Individual Evaluations
Use [`main.py`](src/main.py) to evaluate a single model:
```
python src/main.py --gt-model-path <path_to_gt_model> --est-model-path <path_to_estimated_model>
```

### Absolute Pose Evaluation *(Under Development)*
An **absolute pose evaluation** script is currently in progress. See [`absolute_error_evaluation.py`](src/evaluation/absolute_error_evaluation.py).
- Uses the **Kabsch-Umeyama algorithm** to align estimated and ground truth camera poses.
- Computes **absolute rotation and translation errors**.

## Evaluation Protocol

### Camera Pose Error
The evaluation assesses **relative rotation error (RRE)** and **relative translation error (RTE)**, computed as follows:

#### Formulation
For each image pair \(i\) and \(j\):
```math
  R_{rel} = R_j \cdot R_i^T
```
```math
  t_{rel} = t_j - (R_{rel} \cdot t_i)
```
For implementation details, see [`evaluate_relative_errors`](src/evaluation/relative_error_evaluation.py).

### Novel View Synthesis *(In Progress)*
This protocol evaluates the quality of **novel view synthesis** by comparing rendered images to ground truth images.
- **Nerfstudio** is used to generate novel views via **NeRF** or **Gaussian Splatting**.
- Evaluation is performed using **PSNR** and **SSIM**.

### Current Status
🚧 **This feature is still under development.** 🚧

The script [`run_nerfstudio.py`](src/run_nerfstudio.py) only trains using either **nerfacto** or **splatfacto** methods.
The evaluation part will come later.

#### Running Novel View Synthesis
To train and evaluate novel view synthesis, use:
```
python src/run_nerfstudio.py --dataset-path <PATH_TO_SCENE_IMAGES> --results-path <PATH_TO_RESULTS>
```
- `<PATH_TO_SCENE_IMAGES>`: Path to dataset containing scene images.
- `<PATH_TO_RESULTS>`: Path to SfM method results (e.g., `data/results/glomap/ETH3D/courtyard/colmap/sparse/0`).

📌 **Note:** Files in `<PATH_TO_RESULTS>` must be in **COLMAP format** (`.txt/.bin`).


## TODO
- [x] Implement relative rotation and translation error evaluation
- [ ] Implement absolute pose error evaluation
- [ ] Implement novel view synthesis evaluation


# Resources
- [COLMAP](https://colmap.github.io/)
- [ETH3D](https://www.eth3d.net/)
- [Tanks and Temples](https://www.tanksandtemples.org/)
- [MipNerf](https://jonbarron.info/mipnerf360/)
- [TUM RGB-D SLAM tools](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/tools/)
- [Novel View Synthesis](https://arxiv.org/abs/1601.06950)