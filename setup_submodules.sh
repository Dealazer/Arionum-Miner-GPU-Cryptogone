#!/bin/bash

test -e

echo "Reset submodules"
rm -rf argon2-gpu
git submodule update --init --recursive

echo "# Patch argon2-gpu"
cd argon2-gpu
git apply --reject --whitespace=fix ../argon2-gpu.patch
cp ../kernels/*.cl ./data/kernels/

echo "# Patch argon2"
cd ext/argon2
git apply --whitespace=fix ../../../argon2.patch

cd ../../..
