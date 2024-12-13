# Structure-from-Motion (SfM) Evaluation Protocol

## Overview

This project provides tools for reading, writing, and evaluating Structure-from-Motion (SfM) models.
The evaluation protocol is composed of the following elements:
* Camera pose error
* 3D point cloud error (Maybe not, because it is too difficult to perform alignment)
* Novel view synthesis (https://arxiv.org/abs/1601.06950) (Not implemented yet)

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
- [ ] Check out ATE and RPE for camera pose evaluation. (https://cvg.cit.tum.de/data/datasets/rgbd-dataset/tools/)
  - [ ] Try to do create a 3D point cloud where each point is the camera pose and perform ICP alignment on it.
- [ ] Implement rotation error and report the angle error.
- [ ] Check out the novel view synthesis paper (https://arxiv.org/abs/1601.06950)


# 