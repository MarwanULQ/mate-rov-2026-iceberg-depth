#!/usr/bin/env bash
set -e

# install_prereqs.sh
# Installs all dependencies required for:
# - local video mode
# - RTSP / GStreamer mode
# - live ZED camera mode
# - underwater .npy calibration loading
# - TorchScript model inference support for measure_distance_v2 (manual LibTorch install required)

sudo apt-get update

sudo apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    zlib1g-dev \
    libusb-1.0-0-dev \
    libhidapi-libusb0 \
    libhidapi-dev \
    libopencv-dev \
    libopencv-viz-dev \
    nlohmann-json3-dev \
    gstreamer1.0-tools \
    gstreamer1.0-libav \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev

cd src/iceberg_depth/udev
sudo bash install_udev_rule.sh
cd ../../..

echo "All prerequisites installed successfully."
