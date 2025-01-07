#!/bin/bash

# Export paths
export COLMAP_PATH="/usr/local/bin"
export GOLMAP_PATH="/usr/local/bin"


# Verify COLMAP
if [ ! -x "$COLMAP_PATH" ]; then
  echo "COLMAP executable not found or not executable at: $COLMAP_PATH"
  exit 1
else
  echo "Found COLMAP executable at: ${COLMAP_PATH}"
fi

# Verify GOLMAP
if [ ! -x "$GOLMAP_PATH" ]; then
  echo "GOLMAP executable not found or not executable at: $GOLMAP_PATH"
  exit 1
else
  echo "Found GOLMAP executable at: ${GOLMAP_PATH}"
fi

scene=$1

# Check if the scene and output directory are provided
if [ -z "$scene" ]; then
    echo "Usage: ./run_glomap.sh <scene_dir>"
    exit 1
fi

# Get last part of the scene directory
scene_name=$(basename $scene)
echo "Scene name: $scene_name"

out_dir="results/glomap/$scene_name"
echo "Output directory: $out_dir"

# Check if the scene directory exists
if [ ! -d $scene ]; then
    echo "Scene directory does not exist"
    exit 1
fi

# Check if the output directory exists
if [ ! -d $out_dir ]; then
    mkdir -p $out_dir
fi

# Check if the output directory is not empty and prompt for overwrite
if [ -d "$out_dir" ] && [ "$(ls -A "$out_dir")" ]; then
    echo "Output directory '$out_dir' is not empty. Do you want to overwrite? (y/n)"
    read -r answer
    case "$answer" in
        y|Y)
            echo "Overwriting '$out_dir'..."
            rm -rf "$out_dir"/*  # Clear the directory
            ;;
        n|N)
            echo "Exiting without making changes."
            exit 1
            ;;
        *)
            echo "Invalid input. Exiting."
            exit 1
            ;;
    esac
fi

# Init colmap database 

DATABASE=${out_dir}/sample_reconstruction.db
# If database does not exist, create a new database
if [ ! -f ${DATABASE} ]; then
  ${COLMAP_PATH}/colmap feature_extractor \
    --database_path ${DATABASE} \
    --image_path ${scene}/images \
	--ImageReader.camera_model RADIAL \
	--ImageReader.single_camera 1 \
	--SiftExtraction.use_gpu 1

${COLMAP_PATH}/colmap exhaustive_matcher \
    --database_path ${DATABASE} \
    --SiftMatching.use_gpu 1
fi

# GLOMAP execution

mkdir ${out_dir}/sparse
${GOLMAP_PATH}/glomap mapper \
    --database_path ${DATABASE} \
    --image_path ${scene}/images \
    --output_path ${out_dir}/sparse

# Disable for now 
# ${COLMAP_PATH}/colmap patch_match_stereo \
#     --workspace_path ${out_dir}/dense \
#     --workspace_format COLMAP \
#     --PatchMatchStereo.geom_consistency true

# ${COLMAP_PATH}/colmap stereo_fusion \
#     --workspace_path ${out_dir}/dense \
#     --workspace_format COLMAP \
#     --input_type geometric \
#     --output_path ${out_dir}/dense/fused.ply
