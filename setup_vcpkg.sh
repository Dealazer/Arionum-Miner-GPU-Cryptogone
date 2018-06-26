#!/bin/bash
set -e

echo
echo "- INSTALL VCPKG -"

rm -rf vcpkg

git clone https://github.com/Microsoft/vcpkg
cd vcpkg
./bootstrap-vcpkg.bat
./vcpkg.exe integrate install

