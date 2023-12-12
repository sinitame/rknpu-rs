#!/bin/bash

# Detect script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Get model path
IN_MODEL_PATH=$1
OUT_MODEL_PATH="${IN_MODEL_PATH%.*}.rknn"
touch $OUT_MODEL_PATH

CURRENT_PATH=$(pwd)
CONVERT_SCRIPT="$SCRIPT_DIR/convert.py"
docker run -it \
				-v $IN_MODEL_PATH:/home/model.tflite \
				-v $OUT_MODEL_PATH:/home/model.rknn \
				-v $CONVERT_SCRIPT:/home/convert.py \
				rknn-toolkit \
				python /home/convert.py
