# Structure-from-Motion (SfM) Evaluation Protocol

## Overview
This project provides tools for reading, writing, and evaluating Structure-from-Motion (SfM) models.
The evaluation protocol is composed of the following elements:
* Camera pose error
* 3D triangulation error (Not implemented yet)
* Novel view synthesis (https://arxiv.org/abs/1601.06950) (Not implemented yet)

## Usage
To run the evaluation on all results, run the following command:
```
bash evaluate.sh
```
The script expect the result to be stored in results/<methods>/ETH3D/<scene> directory. For more details, see the script [`evaluate.sh`](evaluate.sh).

To run an individual evaluation, the [`main.py`](Tests/main.py) file in the [`Tests`](Tests) directory can be used.
```
python3 Tests/main.py --gt-model-path <path_to_gt_model> --est-model-path <path_to_estimated_model>
```

An absolute pose evaluation script is **under development** (see [`absolute_error_evaluation.py`](Tests/absolute_error_evaluation.py)).
It first tries to perform alignment between the estimated and ground truth camera poses using the Kabsch-Umeyama algorithm.
Then, it computes the absolute rotation and translation errors.


## Evaluation Protocol
The evaluation protocol assesses the camera relative rotation error (RRE) and relative translation error (RTE). 
Specifically, for each pair of images \(i\) and \(j\), the relative rotation and translation are computed as follows:
```math
  R_{rel} = R_j \cdot R_i^T
```
```math
  t_{rel} = t_j - (R_{rel} \cdot t_i)
```
For more details, see the function [`evaluate_relative_errors`](Tests/relative_error_evaluation.py) in [`evaluation.py`](Tests/relative_error_evaluation.py).

- [ ] Rest is coming soon. (hopefully !)

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
- [ ] Check out the novel view synthesis paper (https://arxiv.org/abs/1601.06950)


# Resources
- [COLMAP](https://colmap.github.io/)
- [ETH3D](https://www.eth3d.net/)
- [Tanks and Temples](https://www.tanksandtemples.org/)
- [MipNerf](https://jonbarron.info/mipnerf360/)
- [TUM RGB-D SLAM tools](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/tools/)
- [Novel View Synthesis](https://arxiv.org/abs/1601.06950)