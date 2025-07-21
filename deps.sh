#!/bin/bash

WORK_DIR="$PWD"
DEPS="$WORK_DIR/deps"

TMP="$WORK_DIR/tmp"

mkdir deps

# git clone https://gitlab.com/libeigen/eigen.git $TMP/eigen
# cd $TMP/eigen
# cmake . -B build -DCMAKE_INSTALL_PREFIX=$DEPS -DCMAKE_BUILD_TYPE=Release
# cmake --install build
# cd ..

###############################################################################
# emp-ot
###############################################################################
mkdir -p $TMP/emp
cd $TMP/emp
export CMAKE_INSTALL_PREFIX="$DEPS"
wget "https://raw.githubusercontent.com/emp-toolkit/emp-readme/master/scripts/install.py"
python install.py --install --tool --ot

###############################################################################
# SEAL
###############################################################################
BUILD_DIR="$DEPS"
DEPS_DIR="$TMP"
git clone https://github.com/microsoft/SEAL.git $DEPS_DIR/SEAL
cd $DEPS_DIR/SEAL
git switch --detach v4.0.0
patch --quiet --no-backup-if-mismatch -N -p1 -i $WORK_DIR/patch/SEAL.patch -d $DEPS_DIR/SEAL/
cmake . -B build -DCMAKE_INSTALL_PREFIX=$BUILD_DIR -DCMAKE_PREFIX_PATH=$BUILD_DIR -DSEAL_USE_MSGSL=OFF -DSEAL_USE_ZLIB=OFF\
	                    -DSEAL_USE_ZSTD=ON -DCMAKE_BUILD_TYPE=Release -DSEAL_USE_INTEL_HEXL=ON -DSEAL_BUILD_DEPS=ON\
                        -DSEAL_THROW_ON_TRANSPARENT_CIPHERTEXT=ON -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build build --target install

rm -rf $TMP
