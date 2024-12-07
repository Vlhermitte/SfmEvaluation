# Structure-from-Motion (SfM) Evaluation Protocol

## Overview

This project provides tools for reading, writing, and evaluating Structure-from-Motion (SfM) models.
The evaluation protocol is composed of the following elements:
* Camera pose error 
* 3D point cloud error
* Novel view synthesis (https://arxiv.org/abs/1601.06950) (Not implemented yet)

## I/O Formats

The project takes as input a COLMAP model and outputs the evaluation results.

## Dataset used
- ETH3D Stereo dataset (https://www.eth3d.net/datasets)
  - Info available are: 
    - Ground truth camera poses
    - Camera intrinsics
    - Ground truth 3D points
- Tanks and Temples dataset (https://www.tanksandtemples.org/download/)
  - Info available are:
    - Ground truth 3D points
- MipNerf dataset (https://jonbarron.info/mipnerf360/)
  - Info available are:
    - TODO: Find out what info is available

### Notes
- In the Tanks and Temples dataset, if the scene is an object (e.g. Ignatius, Truck), the ground truth 3D points \
    is a point cloud of the object without the background.

## TODO
- [ ] Find out what's wrong with the ICP algorithm for the courtyard dataset.
- [ ] Check if we need to perform alignment on the camera pose to evaluate them.
- [ ] Implement rotation error and report the angle error.
- [ ] Check out the novel view synthesis paper (https://arxiv.org/abs/1601.06950)
