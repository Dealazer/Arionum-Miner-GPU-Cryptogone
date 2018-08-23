## How does arionum-gpu-miner v1.4.0 handles the fork ?

For simplicity and stability and in order to release a version in a reasonable time, v1.4.0 behaves like this:
  * during GPU blocks, it will mine at full speed
  * during CPU blocks, it will NOT mine

## Will you implement GPU mining of CPU blocks ?

Yes, but not ETA for now, I will release it when ready, no need to ping me about this on Discord or other social networks.

## What are the strengths of arionum-gpu-miner v1.4.0 ?

Here are the main features:
  * High GPU mining performance
  * Low CPU usage (only 1 thread to manage all GPU jobs)
  * CUDA and OpenCL support
  * Uses cpprestsdk (supports http / https pools)
  * Linux and Windows 10 support

## What is the performance versus Ariominer (Bogdanadnan's miner) ?

Here are performance comparisons of GPU block mining, done on 3 different setups
It is important to note that **CPU mining was disabled in ariominer** (--cpu-intensity 0) this in order to compare only the raw GPU mining speed.

``
-----------------------------------------------------------------------------------------------------------
|       GPU    |          CPU          |      OS     |         ariominer        |    arionum-gpu-miner    |
-----------------------------------------------------------------------------------------------------------
| GTX460, 4GB  | Xeon, 4 cores, 2.4Ghz |  Ubuntu 17  | 250 H/s,  20% CPU usage  |  420 H/s, 7% CPU usage  |
-----------------------------------------------------------------------------------------------------------
| VEGA64, 8GB  | i7, 6 cores, 3.6Ghz   |  Windows 10 | 1270 H/s, 2% CPU usage   |  1880 H/s, 4% CPU usage |
-----------------------------------------------------------------------------------------------------------
| M500M, 2GB   | i7, 4 cores, 2.5Ghz   |  Windows 10 | 56 H/s, 20% CPU usage    |  68 H/s, 5% CPU usage   |
-----------------------------------------------------------------------------------------------------------
``

## Can I use a CPU miner at the same time as the gpu miner ?

Yes you can run any other CPU miner at the same time, just make sure it doesn not uses 100% of your CPU (leave 1 or 2 cores for arionum-pu-miner)
If you use a combined cpu / gpu miner like Ariominer it is recommended to configure it to not mine with GPU (``--gpu-intensity 0``).
If you ever want to use Ariominer to mine CPU blocks with your GPU in parallel of arionum-gpu-miner then reduce the -b value to free some GPU memory for it)

## Can I solo mine with arionum-gpu-miner ?

The only way to solo mine for now is by using https://aro.hashpi.com/ (which is currently not working)
But be aware that because of the 1% fee, there is 1 chance on 100 that your block reward will be taken as fees
I have plans to implement proper solo mining to your own node and with a better fee system in the future.

## What happened with the Arionum fork ?

Multiple forks occured between blocks 80k and 81k, changing ARO mining rules
Now that the fork is stabilized the final rules are : 
  * even blocks are now "CPU blocks"
  * odd blocks are now "GPU blocks"
  * 33% of the block rewards goes to masternodes (you need 100k ARO to run a masternode)

## What is a CPU block ?

A CPU block is exactly the same as the blocks before the fork.
More precisely a CPU block uses argon2i hashing with parameters: time=1,mem=524288,lanes=1
It can be mined with either a GPU or a CPU.
But because a lot of memory (0.5GB) is needed to compute a single hash it makes it hard for GPU to mine those.
Said in another way the cost in Watts to compute a hash with a GPU is way higher than the cost to compute same hash with a CPU.
So it is recommended, but not mandatory to mine CPU blocks with CPUs

## What is a GPU block ?

A GPU block uses argon2i hashing with parameters: time=4,mem=16384,lanes=4
It can be mined with either a CPU or a GPU.
However GPUs are much more faster to compute this kind of blocks (because computation uses few memory and can be parallelized, GPUs like that)
So it is recommended to mine GPU blocks with a GPU.
At the time this FAQ is written, GPU blocks are found so fast that a CPU has almost zero chances to get any share during a GPU block.
(as the mining difficulty adjusts with time GPU blocks will tend to last 4 minutes so this situation will not last)

