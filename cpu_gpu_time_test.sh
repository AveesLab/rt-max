#!/bin/bash

# 기본값 설정
model=""
num_thread=8  # 기본값

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
if [ "$model" == "resnet10" ]; then
    data_file="imagenet1k"
    layer_num=10
elif [ "$model" == "resnet18" ]; then
    data_file="imagenet1k"
    layer_num=18
elif [ "$model" == "yolov2-tiny" ]; then
    data_file="coco"
    layer_num=9
elif [ "$model" == "yolov4-tiny" ]; then
    data_file="coco"
    layer_num=21
elif [ -z "$model" ]; then
    echo "Model not specified. Use -model to specify the model."
    exit 1
else
    echo "Unknown model: $model"
    exit 1
fi

# CPU Layer time
./cpu_layer_time_test.sh -model $model

# GPU Segment time
./gpu_segment_time_test.sh -model $model

# GPU-accel (G[j], G[j]) --> CPU-only execution
for i in {1..8}
do
    echo "(CPU) Running with num_thread=$i"
    for ((j=0; j<layer_num; j++))
    do
        echo "  Testing Gstart=$j Gend=$j"
        ./gpu_accel_test.sh -model ${model} -num_thread $i -Gstart $j -Gend $j -num_exp 30
    done
done