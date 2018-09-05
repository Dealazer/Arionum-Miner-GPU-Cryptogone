#!/bin/bash

set -e

if [ "$1" = "vs2015" ]; then
  vcver="vc14"
elif [ "$1" = "vs2017" ]; then
  vcver="vc15"
elif [ "$1" = "linux" ]; then
  vcver=""
else
  echo "Usage: setup_libs vs2015 / setup_libs vs2017 / setup_libs linux"
  exit 1
fi

echo
echo "- INSTALL ARGON LIB -"
ARGON_PATH=argon2
rm -rf "$ARGON_PATH"
git clone https://bitbucket.org/cryptogone/phc-winner-argon2-for-ario-cpp-miner.git "$ARGON_PATH"
cd "$ARGON_PATH"
git checkout opt_arionum
cd ..

if [ "$1" = "linux" ]; then
  echo "- INSTALL PACKAGES -"
  sudo apt install libgmp-dev python-dev libboost-dev libcpprest-dev zlib1g-dev libssl-dev -y
  exit 0
fi

bash setup_mpir.sh "$vcver"

bash setup_vcpkg.sh

echo
echo "- INSTALL CPPREST -"
./vcpkg/vcpkg.exe install cpprestsdk:x64-windows

echo
echo "- INSTALL BOOST LIBS -"
./vcpkg/vcpkg.exe install boost-algorithm:x64-windows


