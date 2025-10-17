#!/bin/bash

WORK_DIR="$PWD"
DEPS="$WORK_DIR/deps"

TMP="$DEPS/tmp"

BUILD_DIR="$DEPS"
DEPS_DIR="$TMP"

mkdir deps
mkdir $TMP
cd $TMP

# git clone https://gitlab.com/libeigen/eigen.git $TMP/eigen
# cd $TMP/eigen
# cmake . -B build -DCMAKE_INSTALL_PREFIX=$DEPS -DCMAKE_BUILD_TYPE=Release
# cmake --install build

git clone "https://github.com/emp-toolkit/emp-tool.git" emp-tool
cd emp-tool
git checkout 8052d95
sed -i '4i #include <cstdint>' emp-tool/utils/block.h
cmake . -DCMAKE_INSTALL_PREFIX=$DEPS -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build . --target install -j
cd ..

###############################################################################
# emp-ot
###############################################################################
git clone https://github.com/emp-toolkit/emp-ot.git emp-ot
cd emp-ot
git checkout 93b7aa9
cmake $TMP/emp-ot -DCMAKE_INSTALL_PREFIX=$DEPS -DCMAKE_PREFIX_PATH=$DEPS -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build . --target install -j
cd ..

###############################################################################
# SEAL
###############################################################################
git clone https://github.com/microsoft/SEAL.git $DEPS_DIR/SEAL
cd $DEPS_DIR/SEAL
git switch --detach v4.0.0
patch --quiet --no-backup-if-mismatch -N -p1 -i $WORK_DIR/patch/SEAL.patch -d $DEPS_DIR/SEAL/
cmake . -B build -DCMAKE_INSTALL_PREFIX=$BUILD_DIR -DCMAKE_PREFIX_PATH=$BUILD_DIR -DSEAL_USE_MSGSL=OFF -DSEAL_USE_ZLIB=OFF\
	                    -DSEAL_USE_ZSTD=ON -DCMAKE_BUILD_TYPE=Release -DSEAL_USE_INTEL_HEXL=ON -DSEAL_BUILD_DEPS=ON\
                        -DSEAL_THROW_ON_TRANSPARENT_CIPHERTEXT=OFF -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build build --target install --parallel 8

###############################################################################
# troy-nova
###############################################################################
if [[ "$1" = "-gpu" ]]; then
    git clone "https://github.com/lightbulb128/troy-nova.git" $DEPS_DIR/troy-nova
    cd $DEPS_DIR/troy-nova
    git checkout 3354734
    patch --quiet --no-backup-if-mismatch -N -p1 -i $WORK_DIR/patch/troy-nova.patch -d $DEPS_DIR/troy-nova
    bash scripts/build.sh -install -prefix=$BUILD_DIR
fi


rm -rf "$TMP"
