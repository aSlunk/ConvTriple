#!/bin/bash

DEPS="$PWD/deps"

# git clone https://gitlab.com/libeigen/eigen.git
# mkdir deps
# cd eigen
# cmake . -B build -DCMAKE_INSTALL_PREFIX=$DEPS -DCMAKE_BUILD_TYPE=Release
# cmake --install build
# cd ..
# rm -rf eigen

TMP=tmp
mkdir -p $TMP
cd $TMP
export CMAKE_INSTALL_PREFIX="$DEPS"
wget "https://raw.githubusercontent.com/emp-toolkit/emp-readme/master/scripts/install.py"
python install.py --install --tool --ot
cd ..
rm -rf $TMP
