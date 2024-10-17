#!/bin/bash

# CONFIG
ROOT=$(pwd)
LOG_FILE=$(date +"%m%d_%H%M").log
LOG_PATH=${ROOT}/${LOG_FILE}
touch "$LOG_PATH"

# MODEL 설정
MODEL=$1
case "$MODEL" in
    densenet201)
        data_file="imagenet1k"
        layer_num=306
        ;;
    resnet152)
        data_file="imagenet1k"
        layer_num=206
        ;;
    enetb0)
        data_file="imagenet1k"
        layer_num=136
        ;;
    csmobilenet-v2)
        data_file="imagenet1k"
        layer_num=81
        ;;
    squeezenet)
        data_file="imagenet1k"
        layer_num=50
        ;;
    yolov7)
        data_file="coco"
        layer_num=143
        ;;
    yolov7-tiny)
        data_file="coco"
        layer_num=99
        ;;
    yolov4)
        data_file="coco"
        layer_num=162
        ;;
    yolov4-tiny)
        data_file="coco"
        layer_num=38
        ;;
    *)
        echo "Unknown MODEL: $MODEL" | tee -a "$LOG_PATH"
        exit 1
        ;;
esac

DATA=./cfg/${data_file}.data
CFG=./cfg/${MODEL}.cfg
INPUT=data/dog.jpg

num_exp=20
num_thread=6

# 설정을 출력
{
echo "==================EXP 1================="
echo "Experiment Configuration:"
echo "Architecture: GPU-accel"
echo "Model: $MODEL"
echo "Device: CPU"
echo "Number of Experiments: $num_exp"
echo "Number of threads: $num_thread"
echo "Data File: $DATA"
echo "Config File: $CFG"
echo "Input File: $INPUT"
echo "========================================"
} | tee -a "$LOG_PATH"

# 실행 여부를 묻기 (주석 처리된 부분)
# read -p "Do you want to proceed with this configuration? (Y/N): " confirm
# if [[ "$confirm" != "Y" && "$confirm" != "y" ]]; then
#    echo "Experiment aborted." | tee -a "$LOG_PATH"
#    exit 1
# fi

# 실험 실행
{
echo "experiment will be started in 5s"
sleep 1s
echo "4s"
sleep 1s
echo "3s"
sleep 1s
echo "2s"
sleep 1s
echo "1s"
}

cd ~/rt-max

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Experiment Started!!!" | tee -a "$LOG_PATH"
START_SEC=$(date +"%s")

ARCH=data-parallel_nano

for thread in $(seq 1 $num_thread)
do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Start $ARCH architecture with (${thread}-thread)!!" | tee -a "$LOG_PATH"
    START=$(date +%s)
    ./darknet detector $ARCH "$DATA" "$CFG" ./weights/${MODEL}.weights "$INPUT" -num_exp $num_exp -num_thread $thread
    END=$(date +%s)
    elapsed_time=$((END - START))
    hours=$((elapsed_time / 3600))
    minutes=$(((elapsed_time % 3600) / 60))
    seconds=$((elapsed_time % 60))
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] End $ARCH architecture with (${thread}-thread)!!" | tee -a "$LOG_PATH"
    echo "Elapsed time: ${hours}h ${minutes}m ${seconds}s" | tee -a "$LOG_PATH"
done

END_SEC=$(date +"%s")
ELAPSED_TIME=$((END_SEC - START_SEC))
HOURS=$((ELAPSED_TIME / 3600))
MINUTES=$(((ELAPSED_TIME % 3600) / 60))
SECONDS=$((ELAPSED_TIME % 60))
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Experiment Ended!!!" | tee -a "$LOG_PATH"
echo "total Elapsed time: ${HOURS}h ${MINUTES}m ${SECONDS}s" | tee -a "$LOG_PATH"