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
    glayer_max=305
elif [ "$model" == "resnet152" ]; then
    data_file="imagenet1k"
    glayer_max=205
elif [ "$model" == "enetb0" ]; then
    data_file="imagenet1k"
    glayer_max=135
elif [ "$model" == "csmobilenet-v2" ]; then
    data_file="imagenet1k"
    glayer_max=80
elif [ "$model" == "squeezenet" ]; then
    data_file="imagenet1k"
    glayer_max=49
elif [ "$model" == "yolov7" ]; then
    data_file="coco"
    glayer_max=142
elif [ "$model" == "yolov7-tiny" ]; then
    data_file="coco"
    glayer_max=98
elif [ "$model" == "yolov4" ]; then
    data_file="coco"
    glayer_max=161
elif [ "$model" == "yolov4-tiny" ]; then
    data_file="coco"
    glayer_max=37
elif [ -z "$model" ]; then
    echo "Model not specified. Use -model to specify the model."
    exit 1
else
    echo "Unknown model: $model"
    exit 1
fi


# Data_parallel
for var in {1..11}
do
    ./darknet detector data-parallel ./cfg/${data_file}.data ./cfg/${model}.cfg ./weights/${model}.weights data/dog.jpg -num_thread $var -num_exp 30
done

# GPU-accel (GPU 100%)
for var in {1..11}
do
    ./darknet detector gpu-accel ./cfg/${data_file}.data ./cfg/${model}.cfg ./weights/${model}.weights data/dog.jpg -glayer $glayer_max -num_exp 30 -theoretical_exp -theo_thread $var
done
