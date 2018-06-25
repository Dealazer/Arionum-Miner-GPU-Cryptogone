
# Arionum GPU Miner, Windows Edition #

GPU miner for arionum coin : [Arionum](https://www.arionum.com/)

This is a fork of Guli's initial miner, modified in order to make it compile/run on windows
Original sources can be found here: https://bitbucket.org/guli13/arionum-gpu-miner/src

## Updates ##


### 25/06/18 (cryptogone)

* windows first working CUDA & OpenCL versions

## Compilation guide for Windows ###
    Install Visual Studio 2015 (or 2017), Community Edition
    Install CMake for windows
	  https://cmake.org/download/
      add "C:\Program Files (x86)\Windows Kits\10\bin\10.0.17134.0" to path (https://github.com/Microsoft/vcpkg/issues/1689)
      make sure cmake.exe is in the path
    Install CUDA
      https://developer.nvidia.com/cuda-downloads (this also installs OpenCL)
    Install Github for Windows
       all commands below need to be run from a git for Windows console
       git clone https://cryptogone@bitbucket.org/cryptogone/arionum-gpu-miner.git
	   cd arionum-gpu-miner
    Install vcpkg
      ./setup_vcpkg.sh
    Install dependencies
      ./setup_libs.sh
    Init & patch submodules
      ./init_submodules.sh
    Generate Visual Studio solution
      ./gen_prj_vs2015.sh or ./gen_prj_vs2017.sh depending on your visual studio version
    Build
      Open arionum-gpu-miner.sln with visual studio 2015 or 2017
	  In the toolbars, select Release and x64
	  Build Menu -> Build Solution
	  Binaries can be found in the Release/ folder (arionum-cuda-miner.exe / arionum-opencl-miner.exe)

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
    -b, --batchSize=SIZE              batch size [default: 1]
    -t, --threads-per-device=THREADS  thread to use per device [default: 1]
    -?, --help                        show this help and exit

* -b define number of nonces to be hashed in the same batch
* -t define the number of CPU thread for aa GPU device
* -d allow specifying a GPU device
* -u use all GPU devices available

-b and -t are the most important settings, you need to test different values and find the pair giving the best hashrate.
For me it was -b 128 -t 4 !

## Developers Donation

By default the miner will donate 1% of the shares to Guli's address.

    -D parameters allow you to change that: minimum is 0.5%

Donations:

    Guli's wallet: 4hDFRqgFDTjy5okh2A7JwQ3MZM7fGyaqzSZPEKUdgwSM8sKLPEgs8Awpdgo3R54uo1kGMnxujQQpF94qV6SxEjRL
    Cryptogone's wallet: 4bb66RkoTCz63XPBvMyHfWRE1vWif21j77m1kNVNRd7o4YtJdEQWY7BsRVxAYoTdMexSVFGFaekrc3UATTSERwmQ 
