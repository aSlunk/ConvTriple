#!/bin/bash

BUILD_TYPE=Release
BUILD_DIR=build

if [[ ! -d $BUILD_DIR ]]; then
    cmake . -B $BUILD_DIR -DCMAKE_BUILD_TYPE=$BUILD_TYPE -DZLIB_BUILD_EXAMPLES=OFF -DTRIP_VERIFY=ON -DTRIP_COLOR=OFF -DSEAL_USE_MSGSL=OFF -DSEAL_USE_ZLIB=OFF -DSEAL_USE_ZSTD=ON -DSEAL_USE_INTEL_HEXL=ON -DSEAL_THROW_ON_TRANSPARENT_CIPHERTEXT=ON -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DUSE_APPROX_RESHARE=ON
fi

if [[ -n $1 ]]; then
    sed -i 's/\(^#define PROTO\) [0-9]\+/\1 '$1'/' cheetah/defs.hpp
    if [[ -n $2 ]]; then
        sed -i 's/\(^#define COLOR\) [0-9]\+/\1 '$2'/' cheetah/defs.hpp
    fi
fi


cmake --build build
