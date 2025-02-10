#!/bin/bash

# Exit on any error, unset variable, or pipe failure
# set -euo pipefail

# Configuration
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

# Function to verify downloaded file
verify_download() {
    local file=$1
    if [ ! -f "${file}" ] || [ ! -s "${file}" ]; then
        echo "Download verification failed for ${file}. Exiting..."
        return 1  # Exit immediately on failure
    fi
}

# Function to verify extracted files
verify_extraction() {
    local dir=$1
    if [ ! -d "${dir}/images" ] || ! ls "${dir}/images/dslr_images"/*.jpg >/dev/null 2>&1; then
        echo "Extraction verification failed for ${dir}. Exiting..."
        return 1  # Exit immediately on failure
    fi
}

# Function to check if a scene is already downloaded and processed
check_scene_exists() {
    local scene_path=$1
    local image_dir="${scene_path}/images"

    echo "Checking if scene exists: ${scene_path}"

    if [ -d "${image_dir}" ] && ls "${image_dir}"/*.[jJ][pP][gG] >/dev/null 2>&1; then
        local image_count=$(ls "${image_dir}"/*.[jJ][pP][gG] | wc -l)
        echo "Scene already exists with ${image_count} images"
        return 0  # Scene exists
    fi

    echo "Scene does not exist or is incomplete"
    return 1  # Scene does not exist
}

# Function to download and extract a scene
download_and_extract() {
    local scene=$1
    local url="https://www.eth3d.net/data/${scene}_dslr_jpg.7z"
    local output_dir="${BASE_DIR}/" # ${scene}
    local archive_file="${scene}_dslr_jpg.7z"

    echo "Processing scene: ${scene}"

    # Check if the scene already downloaded
    if [ -d "${output_dir}${scene}" ]; then
        echo "Scene already exists. Skipping download..."
        return
    fi

    # Create output directory if it doesn't exist
    mkdir -p "${output_dir}"

    # Download with timeout and retry
    echo "Downloading ${scene}..."
    wget --progress=bar:force --show-progress "${url}" || {
        echo "Failed to download ${scene}"
        return 1  # Exit immediately on failure
    }

    # Verify download
    verify_download "${archive_file}"

    7z x "${archive_file}" -o"${output_dir}" | grep -E "^Extracting|^Everything" || {
        echo "Extraction failed for ${scene}"
        return 1  # Exit immediately on failure
    }
    # Fix permissions
    chmod -R u+w "${output_dir}${scene}"

    # Verify extraction
    # verify_extraction "${output_dir}${scene}"

    # Clean up archive only after successful extraction
    rm "${archive_file}"
    echo "Extracted ${scene}"
}

# Function to move images and clean up folders
organize_images() {
    local scene_path=$1
    local image_dir="${scene_path}/images/dslr_images"
    local dest_dir="${scene_path}/images"

    echo "Organizing images in: ${scene_path}"

    # Ensure destination directory exists
    mkdir -p "${dest_dir}"

    # Check for source files (.jpg or .JPG)
    if ! ls "${image_dir}"/*.JPG >/dev/null 2>&1; then
        echo "No JPG files found in ${image_dir}"
    else
        # Move images with progress feedback
        echo "Moving images..."
        local total_files=$(ls "${image_dir}"/*.JPG | wc -l)
        local current=0

        for img in "${image_dir}"/*.JPG; do
            mv "${img}" "${dest_dir}/" || {
                echo "Failed to move file: ${img}"
                return 1  # Exit immediately on failure
            }
            ((current++))
            printf "\rProgress: [%d/%d] files moved" "${current}" "${total_files}"
        done
        echo

        # Verify all files were moved
        local moved_files=$(ls "${dest_dir}"/*.JPG | wc -l)
        if [ "${moved_files}" -ne "${total_files}" ]; then
            echo "File count mismatch after moving"
            return 1  # Exit immediately on failure
        fi

        # Remove the source directory
        rm -r "${image_dir}"
        echo "Images organized for: ${scene_path}"
    fi
}

# Main execution
main() {
    # Verify wget and 7z are installed
    if ! command -v wget >/dev/null || ! command -v 7z >/dev/null; then
        echo "Required tools (wget and/or 7z) are not installed"
        exit 1
    fi

    # Make sure we are in the root directory of the project
    cd "$(dirname "$0")/.." || exit

    # Create base directory
    mkdir -p "${BASE_DIR}"

    local total_scenes=${#SCENES[@]}
    local current_scene=1
    local skipped_scenes=0
    local successful_scenes=0

    for scene in "${SCENES[@]}"; do
        echo "$scene"
    done

    echo "Starting ETH3D dataset download..."
    echo "Base directory: ${BASE_DIR}"
    echo

    for scene in "${SCENES[@]}"; do
        current_scene="${scene}"  # For cleanup function
        echo "=== Processing scene [${current_scene}/${total_scenes}]: ${scene} ==="

        if check_scene_exists "${BASE_DIR}/${scene}"; then
            echo "Scene already exists. Skipping to the next."
            ((skipped_scenes++))
        else
            # Try to download and process the scene, handling errors gracefully
            if download_and_extract "${scene}" && organize_images "${BASE_DIR}/${scene}"; then
                echo "=== Completed ${scene} ==="
                ((successful_scenes++))
            else
                echo "Failed to process scene: ${scene}. Skipping to the next."
            fi
        fi

        echo
    done

    # Print summary
    echo "=== Download Summary ==="
    echo "Total scenes: ${total_scenes}"
    echo "Already existed: ${skipped_scenes}"
    echo "Successfully downloaded: ${successful_scenes}"
}

# Run the script
main