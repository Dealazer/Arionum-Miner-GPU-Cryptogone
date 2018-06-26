#!/bin/bash

set -e

if [ "$1" = "vs2015" ]; then
  target="Visual Studio 14 2015 Win64"  
elif [ "$1" = "vs2017" ]; then
  target="Visual Studio 15 2017 Win64"
else
  echo "Usage: gen_prj.sh vs2015 / gen_prj.sh vs2017"
  exit 1
fi

pwd=$(pwd)
cmake -D"CMAKE_BUILD_TYPE=Release" -D"CMAKE_TOOLCHAIN_FILE=$pwd/vcpkg/scripts/buildsystems/vcpkg.cmake" -G"$target"

