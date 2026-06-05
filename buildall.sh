#!/bin/bash
# Build all tesseracts in the project
# Assumes tesseract CLI is installed

set -e  # Exit on error

echo "========================================="
echo "Building Tesseracts"
echo "========================================="
echo ""

# Check if tesseract CLI is available
if ! command -v tesseract &> /dev/null; then
    echo "Error: tesseract CLI not found. Please install tesseract-core first."
    echo "Visit: https://github.com/pasteurlabs/tesseract-core"
    exit 1
fi

# Check if tesseract CLI is the correct one
# To avoid potential conflicts with other tools named Tesseract
if ! tesseract --help | grep -q "autodiff"; then
    echo "Error: wrong tesseract CLI. Please install tesseract-core first."
    echo "Visit: https://github.com/pasteurlabs/tesseract-core"
    exit 1
fi


for tess_dir in tesseracts/*/
do
    if [ "${tess_dir}" = "tesseracts/fmpe_posterior/" ] && [ ! -f "${tess_dir}posterior.pkl" ]; then
        echo "Skipping ${tess_dir}: posterior.pkl is missing."
        echo "Train it first with: uv run python fmpe_posterior.py train --n-sims 10000"
        echo ""
        continue
    fi

    echo "Building ${tess_dir}"
    tesseract build ${tess_dir}
    echo "✓ ${tess_dir} built successfully"
    echo ""
done


echo "========================================="
echo "✓ All tesseracts built successfully!"
echo "========================================="
