#!/bin/bash

DEPS="$PWD/deps"

git clone https://gitlab.com/libeigen/eigen.git
mkdir deps
cd eigen
cmake . -B build -DCMAKE_INSTALL_PREFIX=$DEPS -DCMAKE_BUILD_TYPE=Release
cmake --install build
cd ..
rm -rf eigen
