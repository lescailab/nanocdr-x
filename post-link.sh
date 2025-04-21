#!/bin/bash

echo "======================"
echo "DEBUG: Python Version"
python --version
echo "======================"

echo "======================"
echo "DEBUG: Conda List Before Pip Install"
conda list
echo "======================"

echo "Installing dependencies with Pip..."
pip install --no-cache-dir tensorflow==2.15.1 keras==2.15.0 click==8.1.8 pyreadr==0.5.2 tensorflow-metal==1.2.0 || {
    echo "ERROR: Pip failed to install dependencies"
    exit 1
}

echo "======================"
echo "DEBUG: Conda List After Pip Install"
conda list
echo "======================"

echo "post-link.sh completed successfully!"