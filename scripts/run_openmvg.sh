#!/bin/bash

# Check number of input arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: bash run_openmvg.sh <image_dir> <output_dir>"
    echo "Example: bash run_openmvg.sh ./images ./output_dir"
    exit 1
fi

# Input image directory and output directory
IMAGE_DIR=$1
OUTPUT_DIR=$2

# Define intermediate directories
MATCH_DIR="$OUTPUT_DIR/match"
RESULT_DIR="$OUTPUT_DIR/reconstruction"

# Create required directories
mkdir -p "$MATCH_DIR" "$RESULT_DIR"

echo "========================================"
echo "Step 1: Initialize scene (Spherical Camera)"
echo "========================================"
openMVG_main_SfMInit_ImageListing \
    -i "$IMAGE_DIR" \
    -o "$MATCH_DIR" \
    -c 7 \
    -f 1

if [ $? -ne 0 ]; then
    echo "Initialization failed"
    exit 1
fi

echo "========================================"
echo "Step 2: Feature extraction (SIFT)"
echo "========================================"
openMVG_main_ComputeFeatures \
    -i "$MATCH_DIR/sfm_data.json" \
    -o "$MATCH_DIR" \
    -m SIFT \
    -f 1

if [ $? -ne 0 ]; then
    echo "Feature extraction failed"
    exit 1
fi

echo "========================================"
echo "Step 3: Feature matching (FASTCASCADEHASHINGL2)"
echo "========================================"
openMVG_main_ComputeMatches \
    -i "$MATCH_DIR/sfm_data.json" \
    -o "$MATCH_DIR/matches.putative.bin" \
    -n FASTCASCADEHASHINGL2 \
    -r 0.85

if [ $? -ne 0 ]; then
    echo "Feature matching failed"
    exit 1
fi

echo "========================================"
echo "Step 4: Geometric filtering (Fundamental Matrix)"
echo "========================================"
openMVG_main_GeometricFilter \
    -i "$MATCH_DIR/sfm_data.json" \
    -m "$MATCH_DIR/matches.putative.bin" \
    -o "$MATCH_DIR/matches.f.bin" \
    -g a

if [ $? -ne 0 ]; then
    echo "Geometric filtering failed"
    exit 1
fi

echo "========================================"
echo "Step 5: Incremental SfM reconstruction"
echo "========================================"
openMVG_main_SfM \
    -i "$MATCH_DIR/sfm_data.json" \
    -m "$MATCH_DIR" \
    -o "$RESULT_DIR"

if [ $? -ne 0 ]; then
    echo "SfM reconstruction failed"
    exit 1
fi

echo "All steps completed successfully"
echo "Results are saved in: $RESULT_DIR"

echo "========================================"
echo "Step 6: Convert sfm_data.bin to sfm_data.json"
echo "========================================"
openMVG_main_ConvertSfM_DataFormat \
    -i "$RESULT_DIR/sfm_data.bin" \
    -o "$RESULT_DIR/sfm_data.json"

if [ $? -ne 0 ]; then
    echo "Conversion to sfm_data.json failed"
    exit 1
fi

echo "========================================"
echo "Step 7: Export data_views.json"
echo "========================================"
openMVG_main_ConvertSfM_DataFormat \
    -i "$RESULT_DIR/sfm_data.json" \
    -o "$OUTPUT_DIR/data_views.json" \
    -V

if [ $? -ne 0 ]; then
    echo "Exporting data_views.json failed"
    exit 1
fi

echo "========================================"
echo "Step 8: Export data_extrinsics.json"
echo "========================================"
openMVG_main_ConvertSfM_DataFormat \
    -i "$RESULT_DIR/sfm_data.json" \
    -o "$OUTPUT_DIR/data_extrinsics.json" \
    -E

if [ $? -ne 0 ]; then
    echo "Exporting data_extrinsics.json failed"
    exit 1
fi

echo "========================================"
echo "Step 9: Export colored point cloud (pcd.ply)"
echo "========================================"
openMVG_main_ComputeSfM_DataColor \
    -i "$RESULT_DIR/sfm_data.bin" \
    -o "$OUTPUT_DIR/pcd.ply"

if [ $? -ne 0 ]; then
    echo "Exporting pcd.ply failed"
    exit 1
fi

echo "pcd.ply has been saved to: $RESULT_DIR"
