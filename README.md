
# Arionum GPU Miner, Cryptogone Edition #
GPU miner for [Arionum](https://www.arionum.com/)

This is a fork of [Guli's GPU miner](https://bitbucket.org/guli13/arionum-gpu-miner/src), adding windows support and many performance & stability improvements.

Please make sure you install latest Nvidia drivers if you use the CUDA miner !

Miner takes 1% fees (everytime you find a share or block, there is 1 chance on 100 it will be taken as fees).

## Prebuilt Binaries
If you do not want to compile the project by yourself, you can find up to date ready to go binaries at this address :

https://bitbucket.org/cryptogone/arionum-gpu-miner/downloads/

## Simple Ubuntu install config & run

Here is a typical list of shell commands needed to install on Ubuntu (parts starting with a `#` are comments, do not type them !)

I assume this is for a Nvidia card, for AMD card just replace `run_CUDA.sh` with `run_OpenCL.sh` and skip nvidia drivers installation

I also assume this is for Ubuntu 16, if you want to install on another Ubuntu version (17 or 18), 
then use another binary url for the `wget` command, all binaries are [here](https://bitbucket.org/cryptogone/arionum-gpu-miner/downloads/)

    # update list of known packages
    sudo apt-get update

    # install nvidia drivers (only needed if you have a NVidia GPU)
    sudo ubuntu-drivers autoinstall

    # install required packages
    sudo apt-get -y install wget libcpprest libboost-all-dev ocl-icd-opencl-dev

    # create a folder to put the miner in, and go inside it
    mkdir arionum-gpu-miner
    cd arionum-gpu-miner

    # download latest binary archive, make sure that you use the correct url for your Ubuntu version (16, 17, 18 ...)
    wget -O arionum-gpu-miner.tar.gz "https://bitbucket.org/cryptogone/arionum-gpu-miner/downloads/arionum-gpu-miner-v1.5.1-ubuntu16-cuda9.0.tar.gz"

    # extract archive 
    tar xzvf arionum-gpu-miner.tar.gz

    # edit the run script to fit your needs:
    # (set wallet address, worker name, number of batches & threads, eventually change -u to -d if you want to use specific devices)
    # when done, press CTRL+X to exit nano text editor, press Y to save changes
    nano run_CUDA.sh

    # finally you can the miner
    ./run_CUDA.sh

## How to use

1. Either compile yourself or get prebuilt binaries (see above)
2. Uncompress the archive somewhere on your hard drive
3. If you have a Nvidia card then it is better to use the CUDA version otherwise use OpenCL version
4. Edit `run_OpenCL.sh/bat` or `run_CUDA.sh/bat` to set miner id, wallet address and thread / batch count.
5. Launch `run_OpenCL.sh/bat` or `run_CUDA.sh/bat` and enjoy mining ! 

See **Tuning** section below for more advanced usage and choosing thread & batch count.

Also see the [FAQ](https://bitbucket.org/cryptogone/arionum-gpu-miner/src/master/FAQ.md) for more info

## Commandline Options
    -l, --list-devices                     list all available devices and exit
    -d, --devices=devices                  GPU devices to use, examples: -d 0 / -d 1 / -d 0,1,3,5 [default: 0]
    -u, --use-all-devices                  use all available GPU devices (overrides -d)
    -a, --address=address                  public arionum address [default: dev wallet]
    -p, --pool=pool_url                    pool URL [default: http://aropool.com]
    -i, --workerId=worker_id               worker id [default: autogenerated]
    -t, --tasks-per-device=nTaskPerDevice  number of parallel tasks per device, examples: -t 1 / -t 6,3 [default: 1]
    -b, --gpu-batch-size=batchSize         GPU batch size, examples: -b 224 / -b 224,196 [default: 0]
    -k, --cpu-blocks-kernel=kernel_name    kernel for cpu blocks, (shuffle or local_state) [default: local_state if OpenCL, shuffle if CUDA]
    -n, --stats-node=stats_node_url        Programmer Dan stats node url
    -s, --stats-token=MyFancyToken         Programmer Dan stats node secret token
        --skip-cpu-blocks                  do not mine cpu blocks
        --skip-gpu-blocks                  do not mine gpu blocks
        --test-mode                        test CPU/GPU blocks hashrate
        --legacy-5s-hashrate               show 5s avg hashrate instead of last set of batches hashrate
    -?, --help                             show this help and exit

## Tuning performance

First check the list of compute devices on your system:

* For this, launch `listDevices_CUDA.bat/.sh` or `listDevices_OpenCL.bat/.sh`
* If no devices listed, means that CUDA / OpenCL drivers are not properly installed or you have no GPU.

Now decide which GPU device(s) you want to use for mining

* Usually you want all devices, for that use `-u` parameter
* If you want to only use specific devices, list them with -d parameter (ex: `-d 0,3` mines only on devices 0 and 3)
* On laptops combinining a gaming GPU with an Integrated GPU, only mine on the gaming GPU (usually `-d 0`)

On Windows, make sure your virtual memory is bigger than total RAM of all your GPUs

* https://www.geeksinphoenix.com/blog/post/2016/05/10/how-to-manage-windows-10-virtual-memory.aspx
* ex: on a rig with 8x 1080Tis 8GB, windows virtual memory must be greater than 64GB
* this is a limitation of CUDA / OpenCL which need virtual mem to back up GPU memory allocations


Choosing `-t` / `nThreads` and `-b` / `nBatches` values:

* Miner will create `nThreads` GPU tasks per device, each task computing `nBatches` hashes
* Memory used per GPU is `nThreads * nBatches * 0.0167 GB`
* Each GPU also needs an extra `nThreads * 0.05 GB` for CPU/GPU communication & precomputed tables
* In a perfect world, you would just use `-t 1` and set `-b` as high as possible in order to use all available GPU mem
* However GPUs have limitations on the maximum size for a single allocated chunk of memory (usualy around 4GB)
* So if your GPU has more than 4GB you may need to use more threads 
* Also Windows / Linux already uses some of your GPU memory so not all of it is avalaible (use GPUZ or nvidia-smi to see that)
* For example on a Vega64, with `7.5 GB` available over 8, ideally you would use `-t 1 -b 448` (`7.5 / 0.0167 ~= 448`)
* Sadly, doing this will cause miner to crash at launch with an out of memory error (limitation described above)
* So instead use `-t 2 -b 224`, this will allow using the full `2*224*0.0167 ~= 7.5 GB` of memory 
* But, again, reality is more complex than that ... GPUs are complex beasts ;-)
* Some people have reported better hashrates by increasing `nThreads` and adapting `nBatches`accordingly
* For the Vega64 example above it would mean : `-t 3 -b 148` or `-t 4 -b 112` (actually produces lower hashrate for my Vega64)
* Other people have reported better hashrates by **decreasing** `nBatches` while keeping the same `nThreads`
* For the Vega64 example above it would mean for example : `-t 2 -b 200` (actually produces lower hashrate for my Vega64)
* This depends on a lot of factors (GPU model, GPU drivers, OS, overclocking settings etc.)
* So you will have to fiddle a bit, trying various values until you find a good compromise for your mining rig

Test Mode:

* Make sure to use the `--test-mode` parameter to run the miner in test mode when you are tuning `-b / -t`
* Miner will alternate every 30s between CPU and GPU blocks allowing you to quickly see hashrates
* Miner will not submit any share in test mode (so you do not need to specify a pool or wallet address)
* In this mode, the value you want to optimize is the average hashrate (more important than the instant one)
* In test mode you will see that the hashrate may drop a bit every time a block change, this is normal;
* this happens because GPU need to finish current block tasks before launching tasks for the new block
* (with CUDA / OpenCL you cannot easily interrupt a task, you need to wait for it to finish)
* Hashrate loss on block change also explains why in normal mining mode, average hashrate is often lower than instant hashrate

For rigs with many GPUs:

* If you have a rig with more than 2 GPUs, consider running multiple instances of the miner
* for example on a 8 gpu rig you could launch 2 instances of 4 GPU: one with `-d 0,1,2,3`, the other with `-d 4,5,6,7`
* people reported 10 to 20% increase in hashrate by using multiple instances using each 2, 3 or 4 GPUs

## Example -b / -t values

Here are some battle tested values for various GPUs:

    AMD Vega64,           8GB, Win10, OpenCL => -t 2 -b 216
    NVIDIA 1080ti,       11GB, Linux, CUDA   => -t 3 -b 228
    NVIDIA GTX960,        4GB, Linux, CUDA   => -t 1 -b 224
    NVIDIA Quadro M500M,  2GB, Win10, CUDA   => -t 1 -b 104

## Building (Linux)

First, install CUDA SDK (https://developer.nvidia.com/cuda-downloads).
**Important** the CUDA SDK version depends on your GPU drivers (use `nvidia-smi` to view current driver version) :

   driver <  387.26 ==> compatible with CUDA <= 9.0
   driver <  396.14 ==> compatible with CUDA <= 9.1
   driver >= 396.14 ==> compatible with CUDA <= 9.2

Then use the following commands:

    sudo apt-get install git
    git clone https://bitbucket.org/cryptogone/arionum-gpu-miner.git
    cd arionum-gpu-miner
    ./setup_libs.sh linux
    ./setup_submodules.sh
    ./gen_prj.sh linux
    ./make_release_linux.sh

Once done, binaries are put in the `rel/`folder.

Note that `setup_libs.sh` may ask you for your password as it needs to install some packages with `apt-get`

## Building (Windows)

First install the following:

* Visual studio 2015 or 2017 (https://visualstudio.microsoft.com/fr/vs/community/)
* For Visual Studio 2017 make sure to select `Windows 8.1 SDK` during installation
* cuda sdk (https://developer.nvidia.com/cuda-downloads)
* cmake (https://cmake.org/download/), make sure `cmake.exe` is in the system `PATH`
* git for windows (https://git-scm.com/download/win)

Then open a git console and launch the following commands (if needed, replace `vs2015` by `vs2017`)

    git clone https://bitbucket.org/cryptogone/arionum-gpu-miner.git
    cd arionum-gpu-miner
    ./setup_libs.sh vs2015
    ./setup_submodules.sh
    ./gen_prj.sh vs2015
    ./make_release_win.sh

Once done, binaries are put in the `rel/`folder.

You can also skip the last step and instead open `build/arionum-gpu-miner.sln` with Visual Studio 2015 or 2017, then in the toolbar select `Release / x64` then `Build Menu -> Build Solution`.

## Versions release notes

#### Version 1.6.0 (11/22/2018)
* improved `CPU blocks` hashrate for `CUDA` & `OpenCL`
* reduced invalid shares rate (refresh pool info on nonce refusal and try to resubmit nonce on http error)
* support for ProgrammerDan's stats reporting node (new `-n`, `-s` parameters, see FAQ)
* improved `run_CUDA`, `run_OpenCL` scripts
* new `-k` option: allows to choose between `shuffle` kernel and new `local_state` kernel
* `local_state` kernel seems faster on `AMD` devices (default in `arionum-opencl-miner`)
* `shuffle` kernel seems faster on `Nvidia` devices (default in `arionum-cuda-miner`)
* On `AMD` devices, you might need to reduce `-b` value a bit to get the best CPU blocks hashrate
* for example: `-b 216` for v1.6.0 instead of `-b 224` for v1.5.1

#### Version 1.5.1
* show cuda / opencl version used for building in welcome msg
* reduced minimum Cmake version to 3.5 (for Ubuntu 16)
* on Linux, if CUDA sdk not installed will only compile OpenCL miner
* fixed various Linux compilation issues
* improved README (improved linux compilation instructions, added Ubuntu easy setup instructions)

#### Version 1.5.0
* reusing same buffers for CPU & GPU blocks (improves stability, less hashrate loss on block change)
* --test-mode option (benchmarking: in this mode miner alternates GPU/CPU blocks every 20s)
* fixed bug causing small hashrate loss when submitting shares
* new "instant" hashrate computation (less variance), average hashrate not changed
* (use --legacy-5s-hashrate to revert to previous way of computing instant hashrate: avg of last 5s)
* OpenCL program cache (faster startup time when -t > 1)
* linux launch scripts: added option to auto restart the miner if it crashes
* updated FAQ & README

#### Version 1.5.0 beta
* improved CPU block mining hashrate (now in line with Ariominer)
* fixed random crashes at launch
* --skip-cpu-blocks / --skip-gpu-blocks options
* use pinned memory (OpenCL)

#### Version 1.5.0 alpha
* fixed important memory leak
* CPU blocks mining now working 
  (but still way slower than Bogdan's)
* more accurate hashrate computations 
  (the displayed hashrate will probably be a bit lower than before, but more exact)
* now correctly reports CPU/GPU hashrate too pools
* can set worker name via -i parameter (you can also set it via the run_CUDA/run_OpenCL scripts)

#### Version 1.4.0 (08/23/2018)
* block 80k fork support, only mines GPU blocks, idle during CPU ones
* Linux support
* see the [FAQ](https://bitbucket.org/cryptogone/arionum-gpu-miner/src/master/FAQ.md) for more info

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
