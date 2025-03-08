#!/bin/bash

# Exit on any error, unset variable, or pipe failure
# set -euo pipefail

#######################################
# Configuration for ETH3D dataset
#######################################
BASE_DIR="data/datasets/ETH3D"
SCENES=(
    "courtyard"
    "delivery_area"
    "electro"
    "facade"
    "kicker"
    "meadow"
    "office"
    "pipes"
    "playground"
    "relief"
    "relief_2"
    "terrace"
    "terrains"
)

#######################################
# Utility functions
#######################################

# Verify downloaded file exists and is non-empty
verify_download() {
    local file=$1
    if [ ! -f "${file}" ] || [ ! -s "${file}" ]; then
        echo "Download verification failed for ${file}. Exiting..."
        return 1
    fi
}

# Verify extraction for ETH3D scene
verify_extraction() {
    local dir=$1
    if [ ! -d "${dir}/images" ] || ! ls "${dir}/images/dslr_images"/*.jpg >/dev/null 2>&1; then
        echo "Extraction verification failed for ${dir}. Exiting..."
        return 1
    fi
}

# Check if a scene is already downloaded and processed
check_scene_exists() {
    local scene_path=$1
    local image_dir="${scene_path}/images"

    echo "Checking if scene exists: ${scene_path}"
    if [ -d "${image_dir}" ] && ls "${image_dir}"/*.[jJ][pP][gG] >/dev/null 2>&1; then
        local image_count
        image_count=$(ls "${image_dir}"/*.[jJ][pP][gG] | wc -l)
        echo "Scene already exists with ${image_count} images"
        return 0  # Scene exists
    fi

    echo "Scene does not exist or is incomplete"
    return 1  # Scene does not exist
}

#######################################
# ETH3D Functions
#######################################

# Download and extract an ETH3D scene
download_and_extract() {
    local scene=$1
    local url="https://www.eth3d.net/data/${scene}_dslr_jpg.7z"
    local output_dir="${BASE_DIR}/"
    local archive_file="${scene}_dslr_jpg.7z"

    echo "Processing scene: ${scene}"

    # Check if the scene folder already exists
    if [ -d "${output_dir}${scene}" ]; then
        echo "Scene already exists. Skipping download..."
        return
    fi

    # Create output directory if it doesn't exist
    mkdir -p "${output_dir}"

    # Download with progress feedback
    echo "Downloading ${scene}..."
    wget --progress=bar:force --show-progress "${url}" || {
        echo "Failed to download ${scene}"
        return 1
    }

    # Verify the download
    verify_download "${archive_file}"

    # Extract the archive
    7z x "${archive_file}" -o"${output_dir}" | grep -E "^Extracting|^Everything" || {
        echo "Extraction failed for ${scene}"
        return 1
    }

    # Fix permissions
    chmod -R u+w "${output_dir}${scene}"

    # (Optional) Verify extraction if needed
    # verify_extraction "${output_dir}${scene}"

    # Clean up archive only after successful extraction
    rm "${archive_file}"
    echo "Extracted ${scene}"
}

# Move images from nested folder and clean up for ETH3D scene
organize_images() {
    local scene_path=$1
    local image_dir="${scene_path}/images/dslr_images"
    local dest_dir="${scene_path}/images"

    echo "Organizing images in: ${scene_path}"

    # Ensure destination directory exists
    mkdir -p "${dest_dir}"

    # Check for source JPG files (.JPG)
    if ! ls "${image_dir}"/*.JPG >/dev/null 2>&1; then
        echo "No JPG files found in ${image_dir}"
    else
        echo "Moving images..."
        local total_files
        total_files=$(ls "${image_dir}"/*.JPG | wc -l)
        local current=0

        for img in "${image_dir}"/*.JPG; do
            mv "${img}" "${dest_dir}/" || {
                echo "Failed to move file: ${img}"
                return 1
            }
            ((current++))
            printf "\rProgress: [%d/%d] files moved" "${current}" "${total_files}"
        done
        echo

        # Verify all files were moved
        local moved_files
        moved_files=$(ls "${dest_dir}"/*.JPG | wc -l)
        if [ "${moved_files}" -ne "${total_files}" ]; then
            echo "File count mismatch after moving"
            return 1
        fi

        # Remove the now-empty source directory
        rm -r "${image_dir}"
        echo "Images organized for: ${scene_path}"
    fi
}

# Download the complete ETH3D dataset by processing all scenes
download_eth3d() {
    echo "Starting ETH3D dataset download..."
    echo "Base directory: ${BASE_DIR}"
    echo

    mkdir -p "${BASE_DIR}"

    local total_scenes=${#SCENES[@]}
    local skipped_scenes=0
    local successful_scenes=0

    for scene in "${SCENES[@]}"; do
        echo "=== Processing scene: ${scene} ==="
        if check_scene_exists "${BASE_DIR}/${scene}"; then
            echo "Scene already exists. Skipping..."
            ((skipped_scenes++))
        else
            if download_and_extract "${scene}" && organize_images "${BASE_DIR}/${scene}"; then
                echo "=== Completed ${scene} ==="
                ((successful_scenes++))
            else
                echo "Failed to process scene: ${scene}. Skipping to the next."
            fi
        fi
        echo
    done

    echo "=== ETH3D Download Summary ==="
    echo "Total scenes: ${total_scenes}"
    echo "Already existed: ${skipped_scenes}"
    echo "Successfully downloaded: ${successful_scenes}"
}

#######################################
# MipNerf360 Functions
#######################################

download_mipnerf360() {
    local mip_dir="data/datasets/MipNerf360"
    local url="http://storage.googleapis.com/gresearch/refraw360/360_v2.zip"
    local archive_file="360_v2.zip"

    echo "Processing MipNerf360 dataset..."

    # Check if the dataset is already downloaded and extracted
    if [ -d "${mip_dir}" ] && [ "$(ls -A "${mip_dir}")" ]; then
        echo "MipNerf360 dataset already exists. Skipping download..."
        return
    fi

    mkdir -p "${mip_dir}"

    echo "Downloading MipNerf360 dataset..."
    wget --progress=bar:force --show-progress "${url}" || {
        echo "Failed to download MipNerf360 dataset"
        return 1
    }

    # Verify the download
    verify_download "${archive_file}"

    echo "Extracting MipNerf360 dataset..."
    unzip -q "${archive_file}" -d "${mip_dir}" || {
        echo "Extraction failed for MipNerf360 dataset"
        return 1
    }

    # Clean up the archive file
    rm "${archive_file}"
    echo "MipNerf360 dataset downloaded and extracted to ${mip_dir}"
}

#######################################
# LaMAR Functions
#######################################
download_lamar() {
    local lamar_dir="data/datasets/LaMAR"
    local url="https://cvg-data.inf.ethz.ch/lamar/benchmark/"

    echo "Processing LaMAR dataset..."

    # Check if the dataset is already downloaded and extracted
    if [ -d "${lamar_dir}" ] && [ "$(ls -A "${lamar_dir}")" ]; then
        echo "LaMAR dataset already exists. Skipping download..."
        return
    fi

    for scene in "CAB" "HGE" "LIN"; do
      if [ ! -d "$scene.zip" ]; then
        wget --progress=bar:force --show-progress "$url/$scene.zip" || {
          echo "Failed to download LaMAR dataset"
          return 1
        }
      else
        echo "LaMAR dataset already exists. Skipping download..."
      fi
      verify_download "$scene.zip"
      unzip -q "$scene.zip" -d "$lamar_dir" || {
        echo "Extraction failed for LaMAR dataset"
        return 1
      }
      rm "$scene.zip"
    done

    echo "LaMAR dataset downloaded and extracted to ${lamar_dir}"
}


#######################################
# Main Execution
#######################################
main() {
    # Ensure we're in the root directory of the project
    cd "$(dirname "$0")/.." || exit

    echo "Which dataset(s) do you want to download?"
    echo "1) ETH3D dataset"
    echo "2) MipNerf360 dataset"
    echo "3) LaMAR dataset"
    echo "4) All datasets"
    read -rp "Enter your choice (1/2/3/4): " dataset_choice
    echo

    # Check for wget (required for both)
    if ! command -v wget >/dev/null; then
        echo "Error: wget is not installed. Please install wget."
        exit 1
    fi

    case "${dataset_choice}" in
        1)
            # Check for 7z (required for ETH3D)
            if ! command -v 7z >/dev/null; then
                echo "Error: 7z is not installed. Please install 7z."
                exit 1
            fi
            download_eth3d
            ;;
        2)
            # Check for unzip (required for MipNerf360)
            if ! command -v unzip >/dev/null; then
                echo "Error: unzip is not installed. Please install unzip."
                exit 1
            fi
            download_mipnerf360
            ;;
        3)
            # Check for unzip (required for LaMAR)
            if ! command -v unzip >/dev/null; then
                echo "Error: unzip is not installed. Please install unzip."
                exit 1
            fi
            download_lamar
            ;;
        4)
            # Both datasets require their respective tools
            if ! command -v 7z >/dev/null; then
                echo "Error: 7z is not installed. Please install 7z."
                exit 1
            fi
            if ! command -v unzip >/dev/null; then
                echo "Error: unzip is not installed. Please install unzip."
                exit 1
            fi
            download_eth3d
            echo
            download_mipnerf360
            ;;
        *)
            echo "Invalid choice. Exiting."
            exit 1
            ;;
    esac
}

# Run the script
main