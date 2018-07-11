
# Arionum GPU Miner, Windows Edition #
GPU miner for arionum coin : [Arionum](https://www.arionum.com/)

This is a fork of [Guli's GPU miner](https://bitbucket.org/guli13/arionum-gpu-miner/src), porting it to Windows

Please make sure you install latest Nvidia drivers if you use the CUDA miner !

## Updates

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

## Starting the miner

Edit ``run_OpenCL.bat`` or ``run_CUDA.bat`` to your liking with Notepad, save and double click to launch. See sections below for more information.

## Commandline Options

    -l, --list-devices                list all available devices and exit
    -d, --device(s)=DEVICES           GPU devices to use, examples: -d 0 / -d 1 / -d 0,1,3,5 [default: 0]
    -u, --use-all-devices             use all available GPU devices (overrides -d)
    -a, --address=ADDRESS             public arionum address [default: 4hDFRqgFDTjy5okh2A7JwQ3MZM7fGyaqzSZPEKUdgwSM8sKLPEgs8Awpdgo3R54uo1kGMnxujQQpF94qV6SxEjRL]
    -p, --pool=POOL_URL               pool URL [default: http://aropool.com]
    -D, --dev-donation=PERCENTAGE     developer donation [default: 1]
    -t, --threads-per-device=THREADS  number of threads to use per device, ex: -t 1 / -t 6,3 [default: 1]
    -b, --batch-per-device=BATCHES    number of batches to use per device, ex: -b 6 / -b 6,3 [default: 1]
    -?, --help                        show this help and exit

## How to tune options for a good hahsrate

* First check the list of compute devices on your system
  * For this, launch ``listDevices_CUDA.bat`` or ``listDevices_OpenCL.bat``
  * If you see no devices, it means that CUDA / OpenCL drivers are not properly installed or that there are no CUDA/OPENCL devices available
* Now decide which GPU device(s) you want to use for mining 
  * Usually you want all devices, for that use -u parameter (miner will only use GPU devices, skipping CPU devices)
  * If you want to only use specific devices, list them with -d parameter (ex: -d 0,3 mines only on devices 0 and 3)
  * On laptops combinining a gaming GPU with an Integrated GPU, only mine on the gaming GPU (usually -d 0)
* Now let's see how to choose -b and -t
  * for CUDA version:
    * use -t 1
    * for -b use (GPU_MEM_GB - 1) * 2, rounded down
    * ex: GTX960 with 4GB => (4 - 1) * 2 = 6 => -t 1 -b 6 
    * if miner crashes at launch, try reduce the b value of 1 or 2
    * ex: 1080Ti with 11GB does not work with -t 20 but works with -t 18 
  * for OpenCL version:
    * use -t 2
    * for -b use (GPU_MEM_GB - 1), rounded down
    * ex: VEGA64 with 8GB => (8 - 1) = 7 => -t 2 -b 7
    * if miner crashes at launch, try reduce the b value of 1 or 2
* Sample batch files to launch miner are provided: ``run_CUDA.bat`` / ``run_OpenCL.bat``
  * Use them as a basis and change values inside (pool / wallet addresses, -b, -t and either -u or -d parameters)
  
## Example Hashrates

    GTX960     4GB   -b 6  -t 1 => 6.9 H/s
    GTX1080Ti  11GB  -b 18 -t 1 => 35  H/s
    VEGA64     8GB   -b 7  -t 2 => 6.8 H/s	
  
## Developers Donation

By default the miner will donate 1% of the shares to Guli's address.

    -D parameters allow you to change that: minimum is 0.5%

Donations:

    Guli's wallet: 4hDFRqgFDTjy5okh2A7JwQ3MZM7fGyaqzSZPEKUdgwSM8sKLPEgs8Awpdgo3R54uo1kGMnxujQQpF94qV6SxEjRL
    Cryptogone's wallet: 4bb66RkoTCz63XPBvMyHfWRE1vWif21j77m1kNVNRd7o4YtJdEQWY7BsRVxAYoTdMexSVFGFaekrc3UATTSERwmQ 
