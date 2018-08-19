#!/bin/bash

set -e

sudo apt install gcc-6 g++-6 libgmp-dev python-dev libboost-dev libcpprest-dev cmake git make zlib1g-dev libssl-dev libargon2-0-dev -y

# fix for cpprest not finding xlocale
sudo ln -s /usr/include/locale.h /usr/include/xlocale.h

