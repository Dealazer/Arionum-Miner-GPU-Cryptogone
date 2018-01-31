cmake_minimum_required(VERSION 3.9)
project(arionum-gpu-miner)

set(CMAKE_CXX_STANDARD 11)

add_subdirectory(argon2-gpu)

add_executable(arionum-gpu-miner src/arionum-opencl-miner/main.cpp include/miner.h include/openclminer.h src/common/openclminer.cpp include/cudaminer.h src/common/cudaminer.cpp)

target_link_libraries(arionum-gpu-miner
        argon2-opencl
        -lOpenCL
        -lboost_system
        -lssl
        -lcrypto
        -lgmp
        )