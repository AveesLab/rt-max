#!/bin/bash

# 기본값 설정
model=""
num_worker=8
Gstart=0
Gend=1

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

# model 값에 따른 layer_num 및 data_file 설정
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

# CSV 파일 경로 설정
csv_file="./measure/gpu_segments/$model/gpu_segment_partitions_${model}.csv"

# CSV 파일 존재 확인 및 없으면 생성
if [ ! -f "$csv_file" ]; then
    echo "CSV file not found: $csv_file"
    echo "Generating CSV using: python3 gpu_segment_partition_generator.py -model $model -n $layer_num"
    python3 gpu_segment_partition_generator.py -model "$model" -n "$layer_num"

    # 생성 이후에도 존재하지 않으면 오류 처리
    if [ ! -f "$csv_file" ]; then
        echo "Failed to generate CSV file. Check if the generator script ran correctly."
        exit 1
    fi
fi

# CSV 행 개수 가져오기 (헤더 제외)
num_lines=$(tail -n +2 "$csv_file" | wc -l)

# 스레드 수 (1~8) 및 csv 데이터 index (0~num_lines-1)에 따라 반복 실행
for ((thread=1; thread<=8; thread++)); do
    for ((i=0; i<num_lines; i++)); do
        echo "Running: thread=$thread, csv_data=$i"
        ./darknet detector gpu_segment_time ./cfg/${data_file}.data ./cfg/${model}.cfg ./weights/${model}.weights data/dog.jpg \
            -num_thread $thread -num_exp 100 -num_csv_data $i
    done
done
