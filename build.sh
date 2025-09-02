#!/bin/bash

THREADS=8
BUILD_TYPE=Release
BUILD_DIR=build

if [[ ! -d $BUILD_DIR ]]; then
    cmake . -B $BUILD_DIR -DCMAKE_BUILD_TYPE=$BUILD_TYPE -DTRIPLE_VERIFY=OFF -DTRIPLE_COLOR=OFF -DUSE_APPROX_RESHARE=OFF -DTRIPLE_ZERO=ON
fi

if [[ -n $1 ]]; then
    sed -i 's/\(^#define COLOR\) [0-9]\+/\1 '$2'/' src/core/defs.hpp
fi


cmake --build build -j
