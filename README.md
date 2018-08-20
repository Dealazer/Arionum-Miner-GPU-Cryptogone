
# Arionum GPU Miner, Cryptogone Edition #
GPU miner for [Arionum](https://www.arionum.com/)

This is a fork of [Guli's GPU miner](https://bitbucket.org/guli13/arionum-gpu-miner/src), adding windows support and some performance & stability improvements.

Please make sure you install latest Nvidia drivers if you use the CUDA miner !

## Updates

#### Version 1.4.0 (08/20/2018)
* block 80k fork support, only mines GPUS blocks, idle during CPU ones
* Linux support
* see the [FAQ](https://bitbucket.org/cryptogone/rionum-gpu-miner/faq.md) for more info

#### Version 1.3.0 (07/11/2018)
* fixed more OpenCL issues
* switched to a single thread managing all GPUs (improves stability)
* can now set batch size & thread count per device, for example:
*  -d 0 -t 1 -b 6
*  -d 0,1,2 -t 1 -b 6
*  -d 0,1,2 -t 1,2,1 -b 6,3,6
* -k parameter removed
* updated README.md with information on how to get the most out of your GPU(s)

#### Version 1.2.0 (07/02/2018)
* fix OpenCL miner (kernel file was missing in release zip)
* -k parameter now also works for the OpenCL miner
* show error message and stack trace whenever a CUDA or OpenCL exception occurs (will help fixing future issues)

#### Version 1.1.0 (06/27/2018)
* fix CUDA multi gpu support (-d & -u parameters)
* add -k parameter, allows to specify a list of gpu devices to use, ex: -k 0,3,7
* show error message when out of memory during CUDA kernel creation

## Latest Binaries
If you do not want to compile the poject yourself, you can find up to date binaries at this address :

https://bitbucket.org/cryptogone/arionum-gpu-miner/downloads/

## Compilation guide
#### 1. (Windows) Install Visual Studio 2015 or 2017 (Community is sufficient)
Get it at https://visualstudio.microsoft.com/fr/vs/community/

If you install Visual Studio 2017 make sure you also install ```Windows 8.1 SDK``` (you can select it during the install)

#### 2. (Windows) Install CMake
Get it at https://cmake.org/download/
Make sure ```cmake.exe``` is in the system ```PATH```

#### 3. Install CUDA
Get it at https://developer.nvidia.com/cuda-downloads (this also installs ``OpenCL``)

#### 4. Install Git / Git console (Windows)
For Windows, get it at https://git-scm.com/download/win.

For Linux, ``sudo apt-get install git``

**-- (Windows) All commands from here need to be run from a Git for Windows console --**

#### 5. Clone miner repository
    git clone https://bitbucket.org/cryptogone/arionum-gpu-miner.git
    cd arionum-gpu-miner
    
#### 6. Install dependencies 
Run **only one** of those, depending on your system

    ./setup_libs.sh vs2015
    ./setup_libs.sh vs2017
    ./setup_libs.sh linux

Note that on linux this will use ``sudo`` to call ``apt-get`` and so ask for your password for installing packages.

#### 7. Init & patch submodules
    ./setup_submodules.sh
    
#### 8. Generate project
Run **only one** of those, depending on your Visual Studio version

    ./gen_prj.sh vs2015
    ./gen_prj.sh vs2017
    ./gen_prj.sh linux
    
#### 9. Build
On Windows, open ```arionum-gpu-miner.sln``` with Visual Studio 2015 or 2017, then in the toolbar select ```Release / x64``` then ```Build Menu -> Build Solution```.

You can also build from commandline: ``./make_release_win.sh``

On Linux: ``./make_release_linux.sh``.

## Starting the miner using sample scripts

Edit ``run_OpenCL.bat`` or ``run_CUDA.bat`` to your liking with Notepad, save and double click to launch.

On Linux same, but with ``run_OpenCL.sh`` or ``run_CUDA.sh``.

See sections below for more information.

## Commandline Options

    -l, --list-devices                list all available devices and exit
    -d, --device(s)=DEVICES           GPU devices to use, examples: -d 0 / -d 1 / -d 0,1,3,5 [default: 0]
    -u, --use-all-devices             use all available GPU devices (overrides -d)
    -a, --address=ADDRESS             public arionum address [default: 419qwxjJLBRdAVttxcJVT84vdmdq3GP6ghXdQdxN6sqbdr5SBXvPU8bvfVzUXWrjrNrJbAJCvW9JYDWvxenus1pK]
    -p, --pool=POOL_URL               pool URL [default: http://aropool.com]
    -t, --threads-per-device=THREADS  number of threads to use per device, ex: -t 1 / -t 6,3 [default: 1]
    -b, --batch-per-device=BATCHES    number of batches to use per device, ex: -b 6 / -b 6,3 [default: 1]
    -D, --dev-donation=PERCENTAGE     developer donation [default: 1]
    -?, --help                        show this help and exit

## How to tune options for a good hahsrate

1. First check the list of compute devices on your system
    * For this, launch ``listDevices_CUDA.bat/.sh`` or ``listDevices_OpenCL.bat/.sh``
    * If you see no devices, it means that CUDA / OpenCL drivers are not properly installed or that there are no CUDA/OPENCL devices available
2. Now decide which GPU device(s) you want to use for mining
    * Usually you want all devices, for that use -u parameter (miner will only use GPU devices, skipping CPU devices)
    * If you want to only use specific devices, list them with -d parameter (ex: -d 0,3 mines only on devices 0 and 3)
    * On laptops combinining a gaming GPU with an Integrated GPU, only mine on the gaming GPU (usually -d 0)
3. **Choosing -b and -t**
    * for **CUDA** it is recommended to use ``-t 1``
    * for **OpenCL** it is recommended to use ``-t 2``
    * for ``-b`` use an even value like 64,96,128, higher value means better perf but more GPU memory used    
    * Total GPU mem usage of the miner is ~= ``nThreads * nBatches * 0.017 Gb`
    * Usually, not all the memory is available to the miner, so you'll have to fiddle a bit to find the sweet spot for -b
    * if mem usage (influenced by -b and -t) is too high, then miner will crash at launch or will produce bad shares
    * Examples:
        * ``AMD Vega64           8GB, Win10, OpenCL => -t 2 -b 208``
        * ``NVIDIA GTX960        4GB, Linux, CUDA   => -t 1 -b 232``
        * ``NVIDIA Quadro M500M, 2GB, Win10, CUDA   => -t 1 -b 96``
