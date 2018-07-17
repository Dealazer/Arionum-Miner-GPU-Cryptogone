# TODO
* use https://gitlab.com/omos/argon2-gpu, warp-shuffle branch, faster cuda kernel

# Notes about random crashes when using multiple GPUs:
We get this exception: "the launch timed out and was terminated"
https://devtalk.nvidia.com/default/topic/459869/cuda-programming-and-performance/-quot-display-driver-stopped-responding-and-has-recovered-quot-wddm-timeout-detection-and-recovery-/2
https://devtalk.nvidia.com/default/topic/931115/fatal-error-the-launch-timed-out-and-was-terminated-/
https://groups.google.com/forum/#!topic/kaldi-help/DDjNnhnH3jo
https://github.com/fireice-uk/xmr-stak-nvidia/issues/40
https://stackoverflow.com/questions/6185117/cudamemcpy-errorthe-launch-timed-out-and-was-terminated
https://devtalk.nvidia.com/default/topic/483643/cuda-the-launch-timed-out-and-was-terminated/

# Include pathes
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.2\include
argon2-gpu\include
mpir-3.0.0

# Lib pathes
argon2-gpu\Release\
mpir-3.0.0\lib\x64\Release

# Libs
mpir.lib
argon2-cuda.lib or dll
