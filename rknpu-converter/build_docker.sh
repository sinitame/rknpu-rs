#!/bin/bash

set -ex

VERSION="1.5.2"
ASSETS_DIR="$(pwd)/assets"

if ! [ -d $ASSETS_DIR ]; then
    mkdir $ASSETS_DIR
		echo "Downloading Docker file to $ASSETS_DIR .."
		wget -q https://raw.githubusercontent.com/rockchip-linux/rknn-toolkit2/v1.5.2/docker/docker_file/ubuntu_18_04_cp36/Dockerfile_ubuntu_18_04_for_cp36 -P $ASSETS_DIR
		echo "Downloading converter Python wheel to $ASSETS_DIR .."
		wget -q https://raw.githubusercontent.com/rockchip-linux/rknn-toolkit2/v1.5.2/docker/docker_file/ubuntu_18_04_cp36/rknn_toolkit2-1.5.2%2Bb642f30c-cp36-cp36m-linux_x86_64.whl -P $ASSETS_DIR
		echo "Patch Dockerfile .."
		patch -u $ASSETS_DIR/Dockerfile_ubuntu_18_04_for_cp36 -i ./dockerfile.patch
fi

echo "Building docker image.."
cd $ASSETS_DIR
docker build -f $ASSETS_DIR/Dockerfile_ubuntu_18_04_for_cp36 -t rknn-toolkit .

