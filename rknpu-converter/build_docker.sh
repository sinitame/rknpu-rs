#!/bin/bash

set -ex

VERSION="1.6.0"
ASSETS_DIR="$(pwd)/assets"

if ! [ -d $ASSETS_DIR ]; then
    mkdir $ASSETS_DIR
		echo "Downloading Docker file to $ASSETS_DIR .."
		wget -q https://raw.githubusercontent.com/rockchip-linux/rknn-toolkit2/v${VERSION}/docker/docker_file/ubuntu_20_04_cp38/Dockerfile_ubuntu_20_04_for_cp38 -P $ASSETS_DIR
		echo "Downloading converter Python wheel to $ASSETS_DIR .."
		wget -q https://raw.githubusercontent.com/rockchip-linux/rknn-toolkit2/v${VERSION}/docker/docker_file/ubuntu_20_04_cp38/rknn_toolkit2-1.6.0+81f21f4d-cp38-cp38-linux_x86_64.whl -P $ASSETS_DIR
		echo "Patch Dockerfile .."
		patch -u $ASSETS_DIR/Dockerfile_ubuntu_20_04_for_cp38  -i ./dockerfile.patch
fi

echo "Building docker image.."
cd $ASSETS_DIR
docker build -f $ASSETS_DIR/Dockerfile_ubuntu_20_04_for_cp38  -t rknn-toolkit .

