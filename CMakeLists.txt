cmake_minimum_required(VERSION 3.9)
project(arionum-gpu-miner)

set(BINARY_INSTALL_DIR /usr/local/bin)
set(LIBRARY_INSTALL_DIR /usr/local/lib)
set(INCLUDE_INSTALL_DIR /usr/local/include)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CUDA_FOUND FALSE)

if(NOT NO_CUDA)
    find_package(CUDA)
endif()

if(CUDA_FOUND)
    message("INFO: Using CUDA version ${CUDA_VERSION}")
    add_definitions(-DHAVE_CUDA=1)
else()
    message("INFO: Building without CUDA support")
    add_definitions(-DHAVE_CUDA=0)
endif()

if(CUDA_FOUND)
    include_directories(${CUDA_INCLUDE_DIRS})
endif()


add_subdirectory(argon2-gpu)

add_executable(arionum-opencl-miner
        src/arionum-opencl-miner/main.cpp
        include/miner.h
        include/openclminer.h
        src/common/openclminer.cpp
        include/updater.h
        src/common/updater.cpp
        include/stats.h
        src/common/stats.cpp
        include/minerdata.h
        src/common/minerdata.cpp
        include/minersettings.h
        src/common/minersettings.cpp
        src/common/miner.cpp)

target_link_libraries(arionum-opencl-miner
        argon2-opencl
        -lOpenCL
        -lboost_system
        -lssl
        -lcrypto
        -lgmp
        -lpthread
        -lcpprest
        -largon2
        )
if(CUDA_FOUND)
add_executable(arionum-cuda-miner
        src/arionum-cuda-miner/main.cpp
        include/miner.h
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
        src/common/miner.cpp
        include/simplecudaminer.h
        src/common/simplecudaminer.cpp)

target_link_libraries(arionum-cuda-miner
        argon2-cuda
        -lboost_system
        -lssl
        -lcrypto
        -lgmp
        -lpthread
        -lcpprest
        -largon2
        )
endif()
