#!/usr/bin/env bash
set -e

# Build and install the project

cd src/iceberg_depth

mkdir -p build
cd build

cmake ..
make -j"$(nproc)"

sudo make install
sudo ldconfig

