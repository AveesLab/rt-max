#!/bin/bash

# 기본값 설정 (필요한 경우)
model=""

# 파라미터 처리
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -model)
            model="$2"
            shift
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
    shift
done

# model 값에 따른 layer_num 값 설정
if [ "$model" == "densenet201" ]; then
    data_file="imagenet1k"
    layer_num=306
elif [ "$model" == "resnet152" ]; then
    data_file="imagenet1k"
    layer_num=206
elif [ "$model" == "enetb0" ]; then
    data_file="imagenet1k"
    layer_num=136
elif [ "$model" == "csmobilenet-v2" ]; then
    data_file="imagenet1k"
    layer_num=81
elif [ "$model" == "squeezenet" ]; then
    data_file="imagenet1k"
    layer_num=50
elif [ "$model" == "yolov7" ]; then
    data_file="coco"
    layer_num=143
elif [ "$model" == "yolov7-tiny" ]; then
    data_file="coco"
    layer_num=99
elif [ "$model" == "yolov4" ]; then
    data_file="coco"
    layer_num=162
elif [ "$model" == "yolov4-tiny" ]; then
    data_file="coco"
    layer_num=38
elif [ -z "$model" ]; then
    echo "Model not specified. Use -model to specify the model."
    exit 1
else
    echo "Unknown model: $model"
    exit 1
fi

# Data_parallel
./darknet detector data-parallel_sleep ./cfg/${data_file}.data ./cfg/${model}.cfg ./weights/${model}.weights data/dog.jpg -num_thread 11 -num_exp 30

