#!/usr/bin/env bash
set -e

echo "======================"
echo "DEBUG: Python Version"
python --version
echo "======================"

echo "======================"
echo "DEBUG: Pip List Before Install"
python -m pip list
echo "======================"

echo "Installing dependencies with Pip..."

# Base pip dependencies
deps=(
  tensorflow==2.15.1
  keras==2.15.0
  click==8.1.8
  pyreadr==0.5.2
)

# Only install tensorflow-metal on macOS
if [[ "$(uname -s)" == "Darwin" ]]; then
  echo "Detected macOS – adding tensorflow-metal"
  deps+=(tensorflow-metal==1.2.0)
else
  echo "Non‑macOS platform – skipping tensorflow-metal"
fi

# Use python -m pip so we don't depend on a full conda env
python -m pip install --no-cache-dir "${deps[@]}" || {
    echo "ERROR: Pip failed to install dependencies"
    exit 1
}

echo "======================"
echo "DEBUG: Pip List After Install"
python -m pip list
echo "======================"

echo "post-link.sh completed successfully!"