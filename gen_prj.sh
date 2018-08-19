#!/bin/bash

set -e

pwd=$(pwd)

extra="-DCMAKE_TOOLCHAIN_FILE=$pwd/vcpkg/scripts/buildsystems/vcpkg.cmake"
if [ "$1" = "vs2015" ]; then
  target="Visual Studio 14 2015 Win64"  
elif [ "$1" = "vs2017" ]; then
  target="Visual Studio 15 2017 Win64"
elif [ "$1" = "linux" ]; then
  target="Unix Makefiles"
  extra=''
else
  echo "Usage: gen_prj.sh vs2015 / gen_prj.sh vs2017 / gen_prj.sh linux"
  exit 1
fi

mkdir -p build
cd build
cmake -D"CMAKE_BUILD_TYPE=Release" $extra -G"$target" ../
cd ..


