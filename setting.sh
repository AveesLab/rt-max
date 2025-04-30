#!/bin/bash

# chmod +x *.sh

sudo apt update
sudo apt install -y gfortran
sudo apt install -y libgfortran5

./install_blas.sh
./install_blis.sh

cd ~/rt-max
make clean
make -j $(nproc)
./mkdir.sh
