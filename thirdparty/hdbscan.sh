#!/bin/bash
echo "Cloning hdbscan library ...."
git clone https://github.com/ojmakhura/hdbscan.git _hdbscan
cd _hdbscan

echo "Creating the build directory ...."
mkdir build
cd build

echo "Buiding hdbscan ...."
cmake -DCMAKE_INSTALL_PREFIX=../../hdbscan ..
make

echo "installing hdbdscan ...."
make install
cd ../../
pwd
rm -rf _hdbscan
