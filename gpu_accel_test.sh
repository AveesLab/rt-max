#!/bin/bash

# 기본값 설정 (필요한 경우)
model=""
num_worker=10
Gstart=0
Gend=0

# 파라미터 처리
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -model)
            model="$2"
            shift
            ;;
        -worker)
            num_worker="$2"
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

./darknet detector gpu-accel ./cfg/${data_file}.data ./cfg/${model}.cfg ./weights/${model}.weights data/dog.jpg -num_thread $num_worker -Gstart 1 -Gend $layer_num -num_exp 1

# GPU-accelerated with optimal_core
# for ((Gstart=0; Gstart<=layer_num; Gstart++))
# do
#     for ((Gend=Gstart+1; Gend<=layer_num; Gend++))
#     do
#         sleep 3s
#         ./darknet detector gpu-accel ./cfg/${data_file}.data ./cfg/${model}.cfg ./weights/${model}.weights data/dog.jpg -num_thread $num_worker -Gstart $Gstart -Gend $Gend -num_exp 20
#         sleep 3s
#     done
# done
