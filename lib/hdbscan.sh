#!/bin/bash
cd ..
git clone https://github.com/ojmakhura/hdbscan.git

cd hdbscan
mkdir build
cd build
mkdir ../../lib
cmake -DCMAKE_INSTALL_PREFIX=../../lib ..
make
make install
cd ../../

cp -rf hdbscan/modules/gnulib/include lib
cp -rf hdbscan/include/hdbscan lib/include
cp -rf hdbscan/modules/listlib/include/listlib lib/include
rm -rf hdbscan
