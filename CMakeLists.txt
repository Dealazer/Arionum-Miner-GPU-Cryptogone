cmake_minimum_required(VERSION 3.5)
project(arionum-gpu-miner)

set(BINARY_INSTALL_DIR /usr/local/bin)
set(LIBRARY_INSTALL_DIR /usr/local/lib)
set(INCLUDE_INSTALL_DIR /usr/local/include)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CUDA_FOUND FALSE)

if(UNIX)
	set(CMAKE_VERBOSE_MAKEFILE ON)
endif()

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
add_definitions(-DHAVE_CUDA=1)

if (MSVC)
    include_directories(mpir-3.0.0)
    include_directories("vcpkg/installed/x64-windows/include")
    link_directories(mpir-3.0.0/lib/x64/Release)
    link_directories("vcpkg/installed/x64-windows/lib")
endif()

if ( CMAKE_COMPILER_IS_GNUCC )
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wno-ignored-attributes -Wno-conversion-null")
endif()

add_subdirectory(argon2-gpu)

include_directories("argon2/include")

set(COMMON_SRCS)
list(APPEND COMMON_SRCS
    "include/miner_version.h"
    "include/updater.h"
    "include/stats.h"
    "include/minerdata.h"
    "include/minersettings.h"
    "include/miner.h"
    "include/main_common.h"
    "include/timer.h"    
	"src/common/updater.cpp"
    "src/common/stats.cpp"
    "src/common/minerdata.cpp"
    "src/common/minersettings.cpp"
    "src/common/miner.cpp"
    "src/common/timer.cpp"
    "src/common/win_tools.cpp"
    "src/common/testMode.cpp"
    
    "argon2/include/argon2.h"
	"argon2/src/opt.c"
    "argon2/src/argon2.c"
    "argon2/src/core.c"
	"argon2/src/encoding.h"
	"argon2/src/encoding.c"
	"argon2/src/thread.h"
	"argon2/src/thread.c"
	"argon2/src/blake2/blake2b.c"
)

add_executable(arionum-opencl-miner
        ${COMMON_SRCS}
        include/openclminer.h
        src/common/openclminer.cpp
        src/arionum-opencl-miner/main.cpp
)

add_executable(arionum-cuda-miner
        ${COMMON_SRCS}
        include/cudaminer.h
        src/common/cudaminer.cpp
        src/arionum-cuda-miner/main.cpp
)

if(MSVC)
	set(COMMON_LIBS_WIN)
	list(APPEND COMMON_LIBS_WIN
		"argon2"
		"mpir"
	)

	target_link_libraries(arionum-opencl-miner
			${COMMON_LIBS_WIN}
			argon2-opencl
	)

	target_link_libraries(arionum-cuda-miner
			${COMMON_LIBS_WIN}
			argon2-cuda
	)
elseif (UNIX)
	set(COMMON_LIBS_UNIX)
	list(APPEND COMMON_LIBS_UNIX
		"argon2"
		"-lboost_system"
		"-lssl"
		"-lcrypto"
		"-lgmp"
		"-lpthread"
		"-lcpprest"
	)

	target_link_libraries(arionum-opencl-miner
			${COMMON_LIBS_UNIX}
			argon2-opencl
	)

	target_link_libraries(arionum-cuda-miner
			${COMMON_LIBS_UNIX}
			argon2-cuda
	)
endif()
