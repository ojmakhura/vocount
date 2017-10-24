#!/bin/bash
echo "Cloning hdbscan library ...."
git clone https://github.com/ojmakhura/hdbscan.git
cd hdbscan

echo "Creating the build directory ...."
mkdir build
cd build

echo "Buiding hdbscan ...."
cmake -DCMAKE_INSTALL_PREFIX=../.. ..
make

echo "installing hdbdscan ...."
make install
cd ../../
pwd
cp -rf hdbscan/modules/gnulib/include .
cp -rf hdbscan/include/hdbscan ./include
cp -rf hdbscan/modules/listlib/include/listlib ./include
rm -rf hdbscan
