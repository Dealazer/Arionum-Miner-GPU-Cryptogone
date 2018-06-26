#!/bin/bash

test -e

echo "Reset submodules"
rm -rf argon2-gpu
git submodule update --init --recursive

echo "# Patch argon2-gpu"
cd argon2-gpu
git apply ../argon2-gpu.patch

echo "# Patch argon2"
cd ext/argon2
git apply ../../../argon2.patch

cd ../../..
