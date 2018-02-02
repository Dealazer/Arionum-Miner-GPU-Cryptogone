
# README #

GPU miner for arionum coin : [enter link description here](https://www.arionum.com/)

## Installation guide ###

### Ubuntu 17.10
#### Dependencies

    sudo apt install gcc-6 g++-6 libgmp-dev python-dev libboost-dev libcpprest-dev cmake git make zlib1g-dev libssl-dev libargon2-0-dev -y
#### Nvidia GPU - CUDA Installation

    wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.1.85-1_amd64.deb
    sudo dpkg -i cuda-repo-ubuntu1604_9.1.85-1_amd64.deb 
    sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
    sudo apt-get update
    sudo apt-get install cuda -y 
    sudo apt install nvidia-libopencl1-390 -y
    sudo apt install nvidia-opencl-dev -y
#### AMD GPU
You must install AMD opencl drivers.

#### Build the miner

    enter code here
    sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-6 1
    sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-6 1
    git clone https://guli13@bitbucket.org/guli13/arionum-gpu-miner.git
    cd arionum-gpu-miner
    git submodule update --init --recursive
    cmake -DCMAKE_BUILD_TYPE=Release .
    make


### Start miner ###
#### For CUDA

    ./arionum-cuda-miner -p http://aropool.com -a your_address
#### For OpenCL

    ./arionum-opencl-miner -p http://aropool.com -a your_address

