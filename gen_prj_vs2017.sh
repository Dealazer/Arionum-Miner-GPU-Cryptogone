#!/bin/bash

set -e

pwd=$(pwd)

cmake -D"CMAKE_BUILD_TYPE=Release" -G"Visual Studio 15 2017 Win64" -D"CMAKE_TOOLCHAIN_FILE=$pwd/vcpkg/scripts/buildsystems/vcpkg.cmake"
