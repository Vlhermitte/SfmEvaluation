# Structure-from-Motion (SfM) Evaluation Protocol

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Evaluation Protocols](#evaluation-protocols)
  - [Camera Pose Evaluation](#camera-pose-evaluation)
      - [Relative Pose Evaluation](#relative-pose-evaluation)
      - [Absolute Pose Evaluation](#absolute-pose-evaluation-in-progress)
  - [Novel View Synthesis](#novel-view-synthesis-in-progress)
  - [3D Triangulation Evaluation](#3d-triangulation-evaluation)
- [Scripts](#scripts)

## Overview
This project provides tools for reading, writing, and evaluating Structure-from-Motion (SfM) models.
The evaluation protocol consists of the following components:
- **Relative Pose Error**
- **Absolute Pose Error**
- **Novel View Synthesis**
- **3D Triangulation** *(in progress)*

## Installation
To set up the project, execute the following scripts:

### Step 1: Download the Datasets
```bash
bash scripts/get_datasets.sh
```
This script will prompt you to which dataset you want to download. You can choose between:
- **ETH3D**
- **MipNeRF360**
- **LaMAR**
- **All of them**

### Step 2: Clone Required Repositories
```bash
bash scripts/clone_repos.sh
```
This script will clone the following repositories:
- **GLOMAP**
- **VGGSfm**
- **Flowmap**
- **AceZero**

### Step 3: Install Dependencies
Each cloned repository contains a README file with installation instructions.
Follow these steps for each method:
- Most methods require creating a virtual environment and installing dependencies using **conda** or **pip**.
- **GLOMAP** requires compilation or downloading precompiled binaries (Windows only).

To run the Novel View Synthesis evaluation, you need to install [NerfStudio](https://docs.nerf.studio/quickstart/installation.html)

## Evaluation Protocols

### Camera Pose Evaluation
This protocol evaluates the quality of **camera poses** estimated by SfM methods with 2 metrics:
- **Relative Pose Error**
- **Absolute Pose Error**

#### Relative Pose Evaluation
The evaluation assesses **relative rotation error (RRE)** and **relative translation error (RTE)** 
by comparing all pairs of camera poses in the estimated model to the ground truth model.

#### Absolute Pose Evaluation
This protocol evaluates the quality of **absolute camera poses** by comparing estimated poses to ground truth poses.
The estimated model and the ground truth model are aligned using colmap aligner via projection center.

### Running Pose Evaluation
To evaluate relative camera poses, use:
```
python src/run_camera_poses.py --gt-model-path <PATH_TO_GT_MODEL> --est-model-path <PATH_TO_EST_MODEL>
```
- `<PATH_TO_GT_MODEL>`: Path to ground truth model (e.g., `data/ETH3D/courtyard/sparse/0`).
- `<PATH_TO_EST_MODEL>`: Path to estimated model (e.g., `data/results/glomap/ETH3D/courtyard/colmap/sparse/0`).

📌 **Note:** Files in `<PATH_TO_GT_MODEL>` and `<PATH_TO_EST_MODEL>` must be in **COLMAP format** (`.txt/.bin`).

For implementation details, see [`run_camera_poses.py`](src/run_camera_poses.py).

### Novel View Synthesis
This protocol evaluates the quality of **novel view synthesis** by comparing rendered images to ground truth images.
- **Nerfstudio** is used to generate novel views via **NeRF** or **Gaussian Splatting**.
- Evaluation is performed using **PSNR**, **SSIM** and **LPIPS** metrics.

The script [`run_nerfstudio.py`](src/run_nerfstudio.py) only trains using either **nerfacto** or **splatfacto** methods.

#### Running Novel View Synthesis
To train and evaluate novel view synthesis, use:
```
python src/run_nerfstudio.py --dataset-path <PATH_TO_SCENE_IMAGES> --results-path <PATH_TO_RESULTS> --method <METHOD>
```
- `<PATH_TO_SCENE_IMAGES>`: Path to dataset containing scene images.
- `<PATH_TO_RESULTS>`: Path to SfM method results (e.g., `data/results/glomap/ETH3D/courtyard/colmap/sparse/0`).
- `<METHOD> (Optional)`: Method to use for novel view synthesis (`nerfacto` or `splatfacto`). Default is `nerfacto`.

📌 **Note:** Files in `<PATH_TO_RESULTS>` must be in **COLMAP format** (`.txt/.bin`).

### 3D Triangulation Evaluation

This protocol evaluates the quality of 3D triangulation by comparing the reconstructed 3D model to the ground truth scan.

The evaluation can be performed on ETH3D or Tanks and Temples datasets as they provide ground truth scans for each scene.

#### Running 3D Triangulation Evaluation
To evaluate 3D triangulation, use:
```
python src/run_triangulation.py --ref-colmap-path <PATH_TO_REFERENCE_COLMAP_RECONSTRUCTION> --est-colmap-path <PATH_TO_ESTIMATED_RECONSTRUCTION> --gt-pcd-path <PATH_TO_GROUND_TRUTH_PCD> --gt-mlp-path <PATH_TO_GROUND_TRUTH_SCAN>
```

- `<PATH_TO_REFERENCE_COLMAP_RECONSTRUCTION>`: Path to reference COLMAP reconstruction (e.g., `data/ETH3D/courtyard/sparse/0`).
- `<PATH_TO_ESTIMATED_RECONSTRUCTION>`: Path to the estimated reconstruction from your method (e.g., `data/results/glomap/ETH3D/courtyard/colmap/sparse/0`).
- `<PATH_TO_GROUND_TRUTH_PCD>`: Path to the ground truth PCD file (e.g., `data/ETH3D/courtyard/scan.ply`).

Many more parameters are available see [`run_triangulation.py`](src/run_triangulation.py) for details.

The evaluation results, including completeness, accuracy, and F1-scores.

📌 **Note:** 
- The **ETH3D** dataset requires the mlp file to perform pcd alignment.
- The **Tanks and Temples** dataset requires the init_alignment.txt and cropfile to perform pcd alignment and crop the estimated reconstruction.

🚧 **There is still a bug for FlowMap. FlowMap point cloud coordinate system is not coherent with the camera poses, 
making the pcd alignment fail and thus, does not provide valid results for the 3D Triangulation.** 🚧

## Scripts

We provide scripts to run evaluations. However, those script expect the results to be stored in a specific directory structure.
If you decide to save your results in a different structure, you can create your own script or modify the provided scripts.

### Running Evaluation on All Results
To evaluate all **Relative Camera Poses** results, run:
```
bash scripts/camera_poses.sh
```

To evaluate all **Novel View Synthesis** results, run:
```
bash scripts/nerfstudio.sh <METHOD>
```
- `<METHOD> (Optional)`: Method to use for novel view synthesis (`nerfacto` or `splatfacto`). Default is `nerfacto`.

This script will take a long time to run, as it trains a model for each scene and evaluates it.

---
The scripts expect results stored in:
```
data/results/<method>/ETH3D/<scene>/colmap/sparse/0
```
For details, see the scripts in the [`scripts`](scripts) directory.

### Running Individual Evaluations
If you want to run an individual evaluation, see above [Evaluation Protocol](#evaluation-protocol)

---
#### 🚧 Experimental 🚧
We also provide a class that can be used to integrate the evaluation into your own code easily.
It can be used as follows:
```python
from src.evaluator import Evaluator

evaluator = Evaluator()

# Camera poses evaluation
rel_results, abs_results = evaluator.run_camera_evaluator(
    gt_sparse_model="<PATH_TO_GT_MODEL>",
    est_sparse_model="<PATH_TO_EST_MODEL>",
)

# Novel view synthesis evaluation
ssim, psnr, lpips = evaluator.run_novel_view_synthesis_evaluator(
    method="nerfacto", # or "splatfacto"
    dataset_path="<PATH_TO_SCENE_IMAGES>",
    results_path="<PATH_TO_RESULTS>"
)

# Triangulation evaluation (For ETH3D dataset)
tolerances, completenesses, accuracies, f1_scores = evaluator.run_triangulation_evaluator(
    colmap_sparse_ref=...,
    est_sparse_reconstruction=...,
    gt_pcd=...,
    est_pcd=...,
    cropfile=...,
)
```
More details can be found in the [`evaluator.py`](src/evaluator.py) file.

## Done
- [x] Implement relative pose error evaluation
- [x] Implement novel view synthesis evaluation

## TODO
- [ ] Fix absolute pose error evaluation (Maybe try using Procrustes instead of Kabsch)
- [ ] Implement 3D triangulation evaluation (For Tanks and Temples dataset)
- [ ] Use more datasets


# Resources
- [COLMAP](https://colmap.github.io/)
- [ETH3D](https://www.eth3d.net/)
- [Multi-View Stereo Evaluation](https://github.com/ETH3D/multi-view-evaluation)
- [Tanks and Temples](https://www.tanksandtemples.org/)
- [MipNerf](https://jonbarron.info/mipnerf360/)
- [TUM RGB-D SLAM tools](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/tools/)