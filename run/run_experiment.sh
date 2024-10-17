#!/bin/bash

# CONFIG
START_TIME=$(date +"%m-%d %H:%M:%S")
ROOT=$(pwd)
LOG_FILE=$(date +"%m%d_%H%M").log
LOG_PATH=${ROOT}/${LOG_FILE}
touch "$LOG_PATH"

# MODEL 설정
MODEL=densenet201
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
        echo "Unknown MODEL: $MODEL"
        exit 1
        ;;
esac

DATA=./cfg/${data_file}.data
CFG=./cfg/${MODEL}.cfg
INPUT=data/dog.jpg

isGPU=1
case "$isGPU" in
    1)
        device="GPU"
        ;;
    0)
        device="CPU"
        ;;
    *)
        echo "Unknown device: $isGPU"
        exit 1
        ;;
esac

num_exp=40
core_id=4

# 설정을 출력
echo "==================EXP 1================="
echo "Experiment Configuration:"
echo "Architecture: sequential"
echo "Model: $MODEL"
echo "Device: CPU"
echo "Number of Experiments: $num_exp"
echo "Core ID: $core_id"
echo "Data File: $DATA"
echo "Config File: $CFG"
echo "Input File: $INPUT"
echo "========================================"

echo "==================EXP 2================="
echo "Experiment Configuration:"
echo "Architecture: sequential"
echo "Model: $MODEL"
echo "Device: GPU"
echo "Number of Experiments: $num_exp"
echo "Core ID: $core_id"
echo "Data File: $DATA"
echo "Config File: $CFG"
echo "Input File: $INPUT"
echo "========================================"

echo "==================EXP 3================="
echo "Experiment Configuration:"
echo "Architecture: pipeline"
echo "Model: $MODEL"
echo "Device: CPU"
echo "Number of Experiments: $num_exp"
echo "Core ID: $core_id"
echo "Data File: $DATA"
echo "Config File: $CFG"
echo "Input File: $INPUT"
echo "========================================"

echo "==================EXP 4================="
echo "Experiment Configuration:"
echo "Architecture: pipeline"
echo "Model: $MODEL"
echo "Device: GPU"
echo "Number of Experiments: $num_exp"
echo "Core ID: $core_id"
echo "Data File: $DATA"
echo "Config File: $CFG"
echo "Input File: $INPUT"
echo "========================================"
# 실행 여부를 묻기
read -p "Do you want to proceed with this configuration? (Y/N): " confirm
if [[ "$confirm" != "Y" && "$confirm" != "y" ]]; then
    echo "Experiment aborted."
    exit 1
fi

# 실험 실행
echo "experiment will be started in 5s"
sleep 1s
echo "4s"
sleep 1s
echo "3s"
sleep 1s
echo "2s"
sleep 1s
echo "1s"
sleep 1s
cd ~/rt-max

ARCH=sequential
cores=(0 1 2 3 4 5)
isGPU=1
case "$isGPU" in
    1)
        device="GPU"
        ;;
    0)
        device="CPU"
        ;;
    *)
        echo "Unknown device: $isGPU"
        exit 1
        ;;
esac

for core in "${cores[@]}"
do
	echo "[$(date '+%Y-%m-%d %H:%M:%S')] Start $ARCH architecture with $device (${core}-core)!!" | tee -a "$LOG_FILE"
	START=$(date +%s)
	./darknet detector $ARCH "$DATA" "$CFG" ./weights/${MODEL}.weights "$INPUT" -num_exp $num_exp -isGPU $isGPU -core_id $core
	END=$(date +%s)
	elapsed_time=$((END - START))
	hours=$((elapsed_time / 3600))
	minutes=$(((elapsed_time % 3600) / 60))
	seconds=$((elapsed_time % 60))
	echo "[$(date '+%Y-%m-%d %H:%M:%S')] End $ARCH architecture with $device (${core}-core)!!!" | tee -a "$LOG_FILE"
	echo "Elapsed time: ${hours}h ${minutes}m ${seconds}s"
done

isGPU=0
case "$isGPU" in
    1)
        device="GPU"
        ;;
    0)
        device="CPU"
        ;;
    *)
        echo "Unknown device: $isGPU"
        exit 1
        ;;
esac
for core in "${cores[@]}"
do
	echo "[$(date '+%Y-%m-%d %H:%M:%S')] Start $ARCH architecture with $device (${core}-core)!!" | tee -a "$LOG_FILE"
	START=$(date +%s)
	./darknet detector $ARCH "$DATA" "$CFG" ./weights/${MODEL}.weights "$INPUT" -num_exp $num_exp -isGPU $isGPU -core_id $core
	END=$(date +%s)
	elapsed_time=$((END - START))
	hours=$((elapsed_time / 3600))
	minutes=$(((elapsed_time % 3600) / 60))
	seconds=$((elapsed_time % 60))
	echo "[$(date '+%Y-%m-%d %H:%M:%S')] End $ARCH architecture with $device (${core}-core)!!!" | tee -a "$LOG_FILE"
	echo "Elapsed time: ${hours}h ${minutes}m ${seconds}s"
done


ARCH=pipeline
isGPU=1
case "$isGPU" in
    1)
        device="GPU"
        ;;
    0)
        device="CPU"
        ;;
    *)
        echo "Unknown device: $isGPU"
        exit 1
        ;;
esac
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Start $ARCH architecture with $device!!" | tee -a "$LOG_FILE"
START=$(date +%s)
./darknet detector $ARCH "$DATA" "$CFG" ./weights/${MODEL}.weights "$INPUT" -num_exp $num_exp -isGPU $isGPU
END=$(date +%s)
elapsed_time=$((END - START))
hours=$((elapsed_time / 3600))
minutes=$(((elapsed_time % 3600) / 60))
seconds=$((elapsed_time % 60))
echo "[$(date '+%Y-%m-%d %H:%M:%S')] End $ARCH architecture with $device!!!" | tee -a "$LOG_FILE"
echo "Elapsed time: ${hours}h ${minutes}m ${seconds}s"

isGPU=0
case "$isGPU" in
    1)
        device="GPU"
        ;;
    0)
        device="CPU"
        ;;
    *)
        echo "Unknown device: $isGPU"
        exit 1
        ;;
esac
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Start $ARCH architecture with $device!!" | tee -a "$LOG_FILE"
START=$(date +%s)
./darknet detector $ARCH "$DATA" "$CFG" ./weights/${MODEL}.weights "$INPUT" -num_exp $num_exp -isGPU $isGPU
END=$(date +%s)
elapsed_time=$((END - START))
hours=$((elapsed_time / 3600))
minutes=$(((elapsed_time % 3600) / 60))
seconds=$((elapsed_time % 60))
echo "[$(date '+%Y-%m-%d %H:%M:%S')] End $ARCH architecture with $device!!!" | tee -a "$LOG_FILE"
echo "Elapsed time: ${hours}h ${minutes}m ${seconds}s"
