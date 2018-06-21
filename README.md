
# Arionum GPU Miner, Windows Edition #

GPU miner for arionum coin : [Arionum](https://www.arionum.com/)

This is a fork of Guli's initial miner, modified in order to make it compile/run on windows

## Updates ##


### 20/06/18 (cryptogone)

* start windows port

### Update sources guide

    git submodule update --remote
    git pull
    ./make_prj.sh

## Installation guide for Windows ###
    Install Github for Windows
    Install Visual Studio 2015 or 2017 Community Edition
    Install CMake for windows
      add "C:\Program Files (x86)\Windows Kits\10\bin\10.0.17134.0" to path 
      (see https://github.com/Microsoft/vcpkg/issues/1689 for more info)
      make sure cmake.exe is in the path
    Install CUDA
      https://developer.nvidia.com/cuda-downloads (this also installs OpenCL)
    Install vcpkg
      vcpkg (https://github.com/Microsoft/vcpkg)
    Install dependencies
      vcpkg install cpprestsdk cpprestsdk:x64-windows
      vcpkg install boost
      ./setup_mpir.sh
    Prepare submodules
      ./apply_submodules_paches.sh
    Generate Visual Studio solution
      ./gen_prj.sh

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
    
