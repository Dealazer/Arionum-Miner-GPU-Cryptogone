### What are the strengths of arionum-gpu-miner ?

* High performance GPU mining for CPU & GPU blocks
* Low CPU usage (only 1 thread to manage all GPU jobs)
* Supports CUDA and OpenCL
* Linux and Windows 10 support
* Uses cpprestsdk (supports https pools)
* Optimized to lose as less hashrate as possible on block switch
* Supports [ProgrammerDan's stats reporting node 0.2](https://github.com/ProgrammerDan/arionum-node/releases/tag/0.2.0)

### Example settings & Hashrate ?

     --------------------------------------------------------------------------------
    |         Device        |   Params     |     OS      |  CPU block  |  GPU block  |
    |--------------------------------------------------------------------------------|
    | Nvidia M500M, 2GB     | -t 1 -b 104  |  Windows 10 |   11.8 H/s  |   70 H/s    |
    |--------------------------------------------------------------------------------|
    | Nvidia GTX960, 4GB    | -t 1 -b 224  |  Ubuntu 18  |    44 H/s   |  409 H/s    |
    |--------------------------------------------------------------------------------|
    | AMD Vega64 (OC), 8GB  | -t 2 -b 224  |  Windows 10 |  52.5 H/s   |  1970 H/s   |
     --------------------------------------------------------------------------------

Numbers above are from `arionum-cuda-miner` for the NVidia GPUs and `arionum-opencl-miner` for the AMD ones.

It is very important to use the CUDA version on NVidia GPUs: it is always faster or equal to OpenCL.

### Can I use a CPU miner at the same time as the GPU miner ?

Yes you can run any other CPU miner at the same time, just make sure it doesn not uses 100% of your CPU (leave 1 or 2 cores for arionum-gpu-miner)

If you use a combined cpu / gpu miner like ariominer it is recommended to configure it to not mine with GPU (``--gpu-intensity 0``).

It is **not advised** to use another GPU miner at the same time because both miners will compete for GPU ram.

### How to monitor miners with ProgrammerDan's stats reporting node ?

* First, setup an arionum node, for example using [KyleFromOhio's scripts](https://github.com/KyleFromOhio/arionum-scripts)
* Then install [ProgrammerDan's statsAugment](https://github.com/ProgrammerDan/arionum-node/releases) on the node, do not forget to set your **secret token**
* Finally, pass those parameters to the miner: `-n node_url -s secret_token`. 
* For example if node is on same computer as miner: `-n http://127.0.0.1 -s MyFancyToken`


### Why showing multiple hashrates ?

`H/s-last` is the hashrate for the very last set of hashes computed by each GPU. 
This is the 'instant' hashrate at the time the line is shown.

`H/s-5s` (needs `--legacy-5s-hashrate`) is the average hashrate for the last 5 seconds. 
For reasons which are a bit hard to explain (granularity of the GPU tasks) this one has a 
lot of variance and can cause user confusion (this is why I made it optional).

`H/s-avg` is the average hashrate since the miner was started. So this is the "real" value, 
which for example takes into account mining time lost when switching block or anything that might
have slowed down the miner since it has started.

Also note that hashrates are computed separately for GPU & CPU blocks, 
so when you are on a CPU block you see CPU block hashrates and vice versa

### Can I solo mine with arionum-gpu-miner ?

The only way to solo mine for now is by using this pool: https://aro.hashpi.com

But be aware that because of the 1% fee, each time you find a block there is 1 chance on 100 that the full reward will be taken as fees

### What happened with the Arionum fork ?

Multiple forks occured between blocks 80k and 81k, changing ARO mining rules
Now that the fork is stabilized the final rules are : 

  * Even blocks are now "CPU blocks"
  * Odd blocks are now "GPU blocks"
  * 33% of the block rewards goes to masternodes (you need 100k ARO to run a masternode)

### What is a CPU block ?

A CPU block is exactly the same as the blocks before the fork.

More precisely a CPU block uses `argon2i` hashing with parameters: `time=1, mem=524288, lanes=1`

It can be mined with either a GPU or a CPU.

But because a lot of memory (0.5GB) is needed to compute a single hash it makes it hard for GPU to mine those.
(this statement may not be true anymore as ariominer recently managed to use a lot less memory per hash)

Said in another way the cost in Watts to compute a hash with a GPU is way higher than the cost to compute same hash with a CPU.
So it is recommended, but not mandatory to mine CPU blocks with CPUs

### What is a GPU block ?

A GPU block uses `argon2i` hashing with parameters: `time=4, mem=16384, lanes=4`

It can be mined with either a CPU or a GPU.

However GPUs are much more faster to compute this kind of blocks (because computation uses few memory and can be parallelized, GPUs like that)
So it is recommended to mine GPU blocks with a GPU.

