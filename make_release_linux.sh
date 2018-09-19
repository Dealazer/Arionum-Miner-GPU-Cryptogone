#!/bin/bash
set -e

# get version number
VERSION="v$(grep -Po 'MINER_VERSION = "\K[^"]*' include/miner_version.h)"
OS_NAME="linux"
if [ -n "$1" ]; then
	OS_NAME="$1"
fi

OUT_DIR="rel/${VERSION}-${OS_NAME}/"
ARCH_NAME="arionum-gpu-miner-${VERSION}-${OS_NAME}.tar.gz"

echo
echo "-- Building arionum-gpu-miner ${VERSION}-${OS_NAME} --"

# Cleanup
mkdir -p "rel"
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"
rm -f "$ARCH_PATH"

# Build
pushd build
make -j

if ! [ -f "arionum-cuda-miner" ]; then
	echo "Warning: CUDA miner not found (compilation failed or building only OpenCL version)"
fi

if ! [ -f "arionum-opencl-miner" ]; then
	echo "Error: OPENCL miner not found, compilation probably failed..."
	exit 1
fi

# Copy binaries
if [ -f "arionum-cuda-miner" ]; then
	cp arionum-cuda-miner "../$OUT_DIR"
fi
cp arionum-opencl-miner "../$OUT_DIR"
popd

# Copy docs & CL kernels
echo "-- copy docs & kernels --"

cp -r argon2-gpu "$OUT_DIR"
cp README.md "$OUT_DIR"
cp FAQ.md "$OUT_DIR"
if [ -f "arionum-cuda-miner" ]; then
	cp sample_script_files/* "$OUT_DIR"
else
	cp sample_script_files/*CUDA*.sh "$OUT_DIR"
fi

#archive
echo "-- archive --"
pushd "$OUT_DIR"
tar -zcvf "../${ARCH_NAME}" *
popd

# all done !
echo
echo "Build ok, result in $OUT_DIR"

exit 0
