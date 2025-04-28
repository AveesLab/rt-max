#!/bin/bash

# Define the OpenBLAS directory path
OPENBLAS_DIR="$HOME/OpenBLAS"

# Check if the OpenBLAS directory does not exist
if [ ! -d "$OPENBLAS_DIR" ]; then
  # Navigate to the home directory
  cd "$HOME" || exit
  # Clone the OpenBLAS repository
  git clone https://github.com/xianyi/OpenBLAS
  # Navigate to the OpenBLAS directory
  cd OpenBLAS || exit
  # Run the setup script
  ./setup.sh
else
  echo "OpenBLAS directory already exists. Skipping installation."
fi
