#!/bin/bash
set -e

# get version number
VERSION="v$(grep -Po 'MINER_VERSION = "\K[^"]*' include/miner_version.h)"
OS_NAME="win64"
if [ -n "$1" ]; then
	OS_NAME="$1"
fi
OUT_DIR="rel/${VERSION}-${OS_NAME}"
ZIP_PATH="rel/arionum-gpu-miner-${VERSION}-${OS_NAME}.zip"

# get 7zip path
SEVEN_ZIP_X86="${PROGRAMFILES} (x86)/7-Zip/7z.exe"
SEVEN_ZIP_X64="${PROGRAMFILES}/7-Zip/7z.exe"
if [ -f "$SEVEN_ZIP_X86" ]; then
	SEVEN_ZIP="$SEVEN_ZIP_X86"
elif [ -f "$SEVEN_ZIP_X64" ]; then
	SEVEN_ZIP="$SEVEN_ZIP_X64"
else
	echo "Cannot find 7zip, please install it"
	exit 1
fi

#get msbuild (vs2015) path
MSBUILD_VS2015="${PROGRAMFILES} (x86)\\MSBuild\\14.0\\Bin\\MSBuild.exe"
if ! [ -f "$MSBUILD_VS2015" ]; then
	echo "Cannot find msbuild, please install visual studio 2015 (vs2017 not yet supported)"
	exit 1
fi

echo
echo "-- Building arionum-gpu-miner Win64 $VERSION --"

# Cleanup
mkdir -p "rel"
rm -rf "build/Release"
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

rm -f "$ZIP_PATH"

# Clean and Build
"$MSBUILD_VS2015" 'build/arionum-gpu-miner.sln' '//t:Clean' '//p:Configuration=Release' '//p:Platform=x64'
"$MSBUILD_VS2015" 'build/arionum-cuda-miner.vcxproj' '//t:Clean;Build' '//p:Configuration=Release' '//p:Platform=x64'
"$MSBUILD_VS2015" 'build/arionum-opencl-miner.vcxproj' '//t:Clean;Build' '//p:Configuration=Release' '//p:Platform=x64'

# --------- X64: Copy to output folder & zip -------------
if ! [ -f "build/Release/arionum-cuda-miner.exe" ]; then
	echo "Cannot find CUDA miner exe, compilation probably failed..."
	exit 1
fi

if ! [ -f "build/Release/arionum-opencl-miner.exe" ]; then
	echo "Cannot find OPENCL miner exe, compilation probably failed..."
	exit 1
fi

cp build/Release/*.exe "$OUT_DIR"
cp build/Release/*.dll "$OUT_DIR"
unix2dos -n readme.md "$OUT_DIR/readme_${VERSION}.txt"
unix2dos -n faq.md "$OUT_DIR/FAQ_${VERSION}.txt"

mkdir "$OUT_DIR/argon2-gpu"
mkdir "$OUT_DIR/argon2-gpu/data"
mkdir "$OUT_DIR/argon2-gpu/data/kernels"
cp "argon2-gpu/data/kernels/argon2_kernel.cl" "$OUT_DIR/argon2-gpu/data/kernels/argon2_kernel.cl"
cp sample_batch_files/*.bat "$OUT_DIR"
"$SEVEN_ZIP" a -tzip "$ZIP_PATH" "./$OUT_DIR/*"

# all done !
echo
echo "Build ok, result in $OUT_DIR"

exit 0
