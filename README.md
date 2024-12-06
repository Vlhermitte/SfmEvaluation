# Structure-from-Motion (SfM) Evaluation Protocol

## Overview

This project provides tools for reading, writing, and evaluating Structure-from-Motion (SfM) models.
The evaluation protocol is composed of the following elements:
* Camera pose error 
* 3D point cloud error
* Novel view synthesis (https://arxiv.org/abs/1601.06950) (Not implemented yet)

## I/O Formats

The project takes as input a COLMAP model and outputs the evaluation results.


## TODO
- [ ] Find out what's wrong with the ICP algorithm for the courtyard dataset.
- [ ] Fix alignment of the courtyard dataset
- [ ] Check out the novel view synthesis paper (https://arxiv.org/abs/1601.06950)
