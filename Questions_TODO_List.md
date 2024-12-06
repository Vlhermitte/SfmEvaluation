# Questions/Problems and TODO List

## Problems encountered
- The ICP algorithm is working very poorly for the courtyard dataset. The alignment is ver bad. Thus, it is impossible to evaluate the points cloud and camera pose.

## Questions
- The points cloud evaluation is done by finding for each point in the estimated_pcd the closest points in the Ground truth. Is it a good way ?
- The camera pose evaluation is done by aligning the estimated camera pose using the transformation from the ICP alignment. Then it is compared to the ground truth camera pose. Is it a good way ?

- The ETH3D dataset provide a dense points cloud as ground truth. But Tank&Temples dataset provide only a points cloud of the object alone. \
  How Can I perform ICP alignment when the ground is avery narrow point cloud


## TODO
- Fix alignment of the courtyard dataset
- Check out the novel view synthesis paper (https://arxiv.org/abs/1601.06950)
