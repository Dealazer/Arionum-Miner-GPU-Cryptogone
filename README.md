
# Arionum GPU Miner, Windows Edition #

GPU miner for arionum coin : [Arionum](https://www.arionum.com/)

This is a fork of Guli's initial miner, modified in order to make it compile/run on windows

# Miner has been updated with new settings !!!!!

## Updates ##


### 20/06/18 (cryptogone)

* start windows port

### 02/03/18

* FIX for multi GPUs under CUDA devices - now can run on multi GPUs
* FIX miner crash sometimes when updating info or submitting nonce.
* FIX hash reporting to pool
* FIX minor thing, like unused includes
* ADD developer donation parameter, default 1%

### Update guide

    git submodule update --remote
    git pull
    cmake -DCMAKE_BUILD_TYPE=Release
    make

## Installation guide for Windows ###

TBD

## Installation guide for Linux ###

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

##### AMD driver
You must install AMD opencl drivers.
See [AMD site](http://support.amd.com/en-us/kb-articles/Pages/AMDGPU-PRO-Driver-for-Linux-Release-Notes.aspx)

##### Install opencl headers

    sudo apt-get install opencl-headers
    

#### Build the miner

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
    
#### Options

    Options:
    -l, --list-devices                list all available devices and exit
    -u, --use-all-devices             use all available devices
    -a, --address=ADDRESS             public arionum address [default: 4hDFRqgFDTjy5okh2A7JwQ3MZM7fGyaqzSZPEKUdgwSM8sKLPEgs8Awpdgo3R54uo1kGMnxujQQpF94qV6SxEjRL]
    -p, --pool=POOL_URL               pool URL [default: http://aropool.com]
    -d, --device=INDEX                use device with index INDEX [default: 0]
    -b, --batchSize=SIZE              batch size [default: 200]
    -t, --threads-per-device=THREADS  thread to use per device [default: 1]
    -?, --help                        show this help and exit

* -b define number of nonces to be hashed in the same batch
* -t define the number of CPU thread for aa GPU device
* -d allow specifying a GPU device
* -u use all GPU devices available

-b and -t are the most important settings, you need to test different values and find the pair giving the best hashrate.
For me it was -b 128 -t 4 !

## Developer Donation

By default the miner will donate 1% of the shares to my address.

    -D parameters allow you to change that: minimu is 0.5%

If you want to donate directly to support further development, here is my wallet 

    4hDFRqgFDTjy5okh2A7JwQ3MZM7fGyaqzSZPEKUdgwSM8sKLPEgs8Awpdgo3R54uo1kGMnxujQQpF94qV6SxEjRL
    
