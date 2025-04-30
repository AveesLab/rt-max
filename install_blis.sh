#!/bin/bash

# Define the BLIS directory path
BLIS_DIR="$HOME/blis"

# Check if the BLIS directory does not exist
if [ ! -d "$BLIS_DIR" ]; then
  # Update package lists
  sudo apt-get update
  # Install build-essential and libblis-dev packages
  sudo apt-get install -y build-essential libblis-dev
  # Clone the BLIS repository
  git clone https://github.com/flame/blis.git "$BLIS_DIR"
  # Navigate to the BLIS directory
  cd "$BLIS_DIR" || exit
  # Configure the build
  ./configure auto
  # Build BLIS
  make
  # Install BLIS
  sudo make install
  # export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
else
  echo "BLIS directory already exists. Skipping installation."
fi

