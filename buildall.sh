#!/usr/bin/env bash
# Build every locally configured Tesseract image.

set -euo pipefail

printf '%s\n' "Building Tesseracts" "==================="

if ! command -v tesseract >/dev/null 2>&1; then
    printf '%s\n' \
        "Error: tesseract CLI not found. Install tesseract-core first." \
        "https://github.com/pasteurlabs/tesseract-core"
    exit 1
fi

if ! tesseract --version >/dev/null 2>&1; then
    printf '%s\n' \
        "Error: the installed tesseract command could not report its version." \
        "https://github.com/pasteurlabs/tesseract-core"
    exit 1
fi

for tess_dir in tesseracts/*/; do
    if [ "${tess_dir}" = "tesseracts/fmpe_posterior/" ] && [ ! -f "${tess_dir}posterior.pkl" ]; then
        printf 'Skipping %s: posterior.pkl is missing.\n' "${tess_dir}"
        printf '%s\n\n' "Train it first with: make train-posterior"
        continue
    fi

    printf 'Building %s\n' "${tess_dir}"
    tesseract build "${tess_dir}"
    printf 'Built %s\n\n' "${tess_dir}"
done

printf '%s\n' "All available Tesseracts built successfully."
