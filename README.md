
# Arionum GPU Miner, Windows Edition #
GPU miner for arionum coin : [Arionum](https://www.arionum.com/)

This is a fork of [Guli's GPU miner](https://bitbucket.org/guli13/arionum-gpu-miner/src), porting it to Windows

## Compilation guide for Windows
#### 1. Install Visual Studio 2015 or 2017 (Community)
Get it at https://visualstudio.microsoft.com/fr/vs/community/

If you install Visual Studio 2017 make sure you also install ```Windows 8.1 SDK``` (you can select it during the install)

#### 2. Install CMake
Get it at https://cmake.org/download/

Make sure ```cmake.exe``` is in the system ```PATH```

#### 3. Install CUDA
Get it at https://developer.nvidia.com/cuda-downloads (this also installs ``OpenCL``)

#### 4. Install Git / Git console
Get it at https://git-scm.com/download/win

**All commands from here need to be run from a Git for Windows console**

#### 5. Clone miner repository
    git clone https://bitbucket.org/cryptogone/arionum-gpu-miner.git
    cd arionum-gpu-miner
	
#### 6. Install dependencies 
Run **only one** of those, depending on your Visual Studio version

    ./setup_libs.sh vs2015
    ./setup_libs.sh vs2017
	
#### 7. Init & patch submodules
    ./setup_submodules.sh
	
#### 8. Generate Visual Studio solution
Run **only one** of those, depending on your Visual Studio version

    ./gen_prj.sh vs2015
    ./gen_prj.sh vs2017
	
#### 9. Build
Open ```arionum-gpu-miner.sln``` with Visual Studio 2015 or 2017 

In the toolbar select ```Release / x64``` then ```Build Menu -> Build Solution```

#### 10. Binaries 
After a succesfull build, binaries can be found here:

    Release/arionum-cuda-miner.exe
    Release/arionum-opencl-miner.exe

## Start miner

#### For CUDA

    ./arionum-cuda-miner -p http://aropool.com -a your_address

#### For OpenCL

    ./arionum-opencl-miner -p http://aropool.com -a your_address
    
#### Options

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
For Guli it was -b 128 -t 4 (GPU unknown), Cryptogone tested on a GTX960 and uses -b 6 -t 1 (6.9 H/s)

If miner crashes at launch this is probably because batch size or number of threads are too big relative to your GPU RAM size.

## Developers Donation

By default the miner will donate 1% of the shares to Guli's address.

    -D parameters allow you to change that: minimum is 0.5%

Donations:

    Guli's wallet: 4hDFRqgFDTjy5okh2A7JwQ3MZM7fGyaqzSZPEKUdgwSM8sKLPEgs8Awpdgo3R54uo1kGMnxujQQpF94qV6SxEjRL
    Cryptogone's wallet: 4bb66RkoTCz63XPBvMyHfWRE1vWif21j77m1kNVNRd7o4YtJdEQWY7BsRVxAYoTdMexSVFGFaekrc3UATTSERwmQ 
