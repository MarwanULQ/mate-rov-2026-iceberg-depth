#!/usr/bin/env bash
set -e

# Build and install the project

cd src/iceberg_depth

mkdir -p build
if [[ ! -w build ]]; then
	echo "Build directory is not writable: src/iceberg_depth/build"
	echo "Fix with: sudo chown -R \"$USER\":\"$USER\" src/iceberg_depth/build"
	exit 1
fi
cd build

CMAKE_ARGS=()

if [[ -n "${LIBTORCH_ROOT:-}" ]]; then
	CMAKE_ARGS+=("-DBUILD_TORCH_EXAMPLE=ON" "-DLIBTORCH_ROOT=${LIBTORCH_ROOT}" "-DCMAKE_CUDA_ARCHITECTURES=native")
	echo "Using LibTorch root: ${LIBTORCH_ROOT}"
elif [[ -d /home/mar0/ROV/Length/third_party/libtorch ]]; then
	CMAKE_ARGS+=("-DBUILD_TORCH_EXAMPLE=ON" "-DLIBTORCH_ROOT=/home/mar0/ROV/Length/third_party/libtorch" "-DCMAKE_CUDA_ARCHITECTURES=native")
	echo "Using LibTorch root: /home/mar0/ROV/Length/third_party/libtorch"
fi

if [[ -x /usr/local/cuda/bin/nvcc ]]; then
	CMAKE_ARGS+=("-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc")
fi

cmake .. "${CMAKE_ARGS[@]}"
make -j"$(nproc)"

sudo make install
sudo ldconfig

