#!/bin/bash

set -e

if [ "$1" = "vs2015" ]; then
  vcver="vc14"
elif [ "$1" = "vs2017" ]; then
  vcver="vc15"
elif [ "$1" = "linux" ]; then
  # maybe too much things here...
  sudo apt install libgmp-dev python-dev libboost-dev libcpprest-dev git make zlib1g-dev libssl-dev -y
  exit 0
else
  echo "Usage: setup_libs vs2015 / setup_libs vs2017 / setup_libs linux"
  exit 1
fi

bash setup_mpir.sh "$vcver"

bash setup_vcpkg.sh

echo
echo "- INSTALL CPPREST -"
./vcpkg/vcpkg.exe install cpprestsdk:x64-windows

echo
echo "- INSTALL BOOST LIBS -"
./vcpkg/vcpkg.exe install boost-algorithm:x64-windows
