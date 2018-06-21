#!/bin/bash

test -e

cd argon2-gpu
git diff --ignore-submodules > ../argon2-gpu.patch

cd ext/argon2
git diff > ../../../argon2.patch

cd ../../..
