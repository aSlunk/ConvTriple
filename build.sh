#!/bin/bash

BUILD_TYPE=Release
BUILD_DIR=build

if [[ ! -d $BUILD_DIR ]]; then
    cmake . -B $BUILD_DIR -DCMAKE_BUILD_TYPE=$BUILD_TYPE -DZLIB_BUILD_EXAMPLES=OFF -DTRIP_VERIFY=OFF -DTRIP_COLOR=OFF
fi

if [[ -n $1 ]]; then
    sed -i 's/\(^#define PROTO\) [0-9]\+/\1 '$1'/' cheetah/defs.hpp
    if [[ -n $2 ]]; then
        sed -i 's/\(^#define COLOR\) [0-9]\+/\1 '$2'/' cheetah/defs.hpp
    fi
fi


cmake --build build
