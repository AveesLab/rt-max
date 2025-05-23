#!/bin/bash

# 기본값 설정 (필요한 경우)
model=""
num_thread=8
Gstart=0
Gend=1
num_exp=30

# 파라미터 처리
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -model)
            model="$2"
            shift
            ;;
        -num_thread)
            num_thread="$2"
            shift
            ;;
        -Gstart)
            Gstart="$2"
            shift
            ;;
        -Gend)
            Gend="$2"
            shift
            ;;
        -num_exp)
            num_exp="$2"
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
if [ "$model" == "resnet10" ]; then
    data_file="imagenet1k"
    layer_num=11
elif [ "$model" == "resnet18" ]; then
    data_file="imagenet1k"
    layer_num=19
elif [ "$model" == "yolov2-tiny" ]; then
    data_file="coco"
    layer_num=10
elif [ "$model" == "yolov4-tiny" ]; then
    data_file="coco"
    layer_num=22
elif [ -z "$model" ]; then
    echo "Model not specified. Use -model to specify the model."
    exit 1
else
    echo "Unknown model: $model"
    exit 1
fi

./darknet detector gpu-accel ./cfg/${data_file}.data ./cfg/${model}.cfg ./weights/${model}.weights data/dog.jpg -num_thread $num_thread -num_exp $num_exp -Gstart $Gstart -Gend $Gend