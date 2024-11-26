# Structure-from-Motion (SfM) Evaluation Protocol

## Overview

This project provides tools for reading, writing, and evaluating Structure-from-Motion (SfM) models.
The evaluation protocol is composed of the following elements:
* Camera pose error (evaluating the estimated camera rotations and translations)
* Camera intrinsics error (if available in the dataset)
* 3D sparse point cloud error

## I/O Formats

The project takes as input a COLMAP model and outputs the evaluation results.


## TODO

- [ ] Code ICP alignment for point cloud alignment
- [ ] Perform 3D sparse point cloud error evaluation (after ICP alignment)
- [ ] Perform camera pose error evaluation
- [ ] Perform camera intrinsics error evaluation
- [ ] Check this repo : https://github.com/ETH3D/multi-view-evaluation?tab=readme-ov-file
