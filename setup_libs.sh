#!/bin/bash

set -e

if [ "$1" = "vs2015" ]; then
  vcver="vc14"
elif [ "$1" = "vs2017" ]; then
  vcver="vc15"
else
  echo "Usage: setup_libs vs2015 / setup_libs vs2017"
  exit 1
fi

bash setup_mpir.sh "$vcver"

bash setup_vcpkg.sh

echo
echo "- INSTALL CPPREST -"
./vcpkg/vcpkg.exe install cpprestsdk:x64-windows

echo
echo "- INSTALL BOOST-ALGORITHM -"
./vcpkg/vcpkg.exe install boost-algorithm:x64-windows