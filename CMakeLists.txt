cmake_minimum_required(VERSION 3.9)
project(arionum-gpu-miner)

set(BINARY_INSTALL_DIR /usr/local/bin)
set(LIBRARY_INSTALL_DIR /usr/local/lib)
set(INCLUDE_INSTALL_DIR /usr/local/include)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(argon2-gpu)

add_executable(arionum-gpu-miner
        src/arionum-opencl-miner/main.cpp
        include/miner.h
        include/openclminer.h
        src/common/openclminer.cpp
        include/cudaminer.h
        src/common/cudaminer.cpp
        include/updater.h
        src/common/updater.cpp
        include/stats.h
        src/common/stats.cpp
        include/minerdata.h
        src/common/minerdata.cpp
        include/minersettings.h
        src/common/minersettings.cpp
        src/common/miner.cpp)

target_link_libraries(arionum-gpu-miner
        argon2-opencl
        -lOpenCL
        -lboost_system
        -lssl
        -lcrypto
        -lgmp
        -lpthread
        -lcpprest
        )