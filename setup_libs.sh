#!/bin/bash

set -e

bash setup_mpir.sh

bash setup_vcpkg.sh

./vcpkg/vcpkg.exe install cpprestsdk:x64-windows boost-algorithm:x64-windows
