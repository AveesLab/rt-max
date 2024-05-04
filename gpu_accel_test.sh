#!/bin/bash
# 평균을 계산하고 반환하는 함수
calculate_average_int() {
    local file_path=$1
    local column=$2

    # 파일 존재 여부 확인
    if [ ! -f "$file_path" ]; then
        echo "Error: File '$file_path' does not exist."
        return 1  # 에러 상태 반환
    fi

    # 주어진 열의 값 추출 및 평균 계산, 결과 반환
    awk -v col="$column" -F ',' '
    BEGIN {
        sum = 0
        count = 0
    }
    {
        if (NR > 1 && $col != "") { # 첫 번째 행을 제외하고, 비어 있지 않은 값을 처리
            sum += $col
            count++
        }
    }
    END {
        if (count > 0) {
            printf "%d", sum / count  # 평균을 출력하지 않고 printf를 통해 형식을 지정하여 반환
        } else {
            print "NaN"  # 데이터가 없는 경우 NaN 반환
        }
    }' "$file_path"
}

calculate_average_float() {
    local file_path=$1
    local column=$2

    # 파일 존재 여부 확인
    if [ ! -f "$file_path" ]; then
        echo "Error: File '$file_path' does not exist."
        return 1  # 에러 상태 반환
    fi

    # 주어진 열의 값 추출 및 평균 계산, 결과 반환
    awk -v col="$column" -F ',' '
    BEGIN {
        sum = 0
        count = 0
    }
    {
        if (NR > 1 && $col != "") { # 첫 번째 행을 제외하고, 비어 있지 않은 값을 처리
            sum += $col
            count++
        }
    }
    END {
        if (count > 0) {
            printf "%f", sum / count  # 평균을 출력하지 않고 printf를 통해 형식을 지정하여 반환
        } else {
            print "NaN"  # 데이터가 없는 경우 NaN 반환
        }
    }' "$file_path"
}

# 기본값 설정 (필요한 경우)
model=""
clean_mode=false  # 'clean' 모드를 위한 변수 추가

# 파라미터 처리
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -model)
            model="$2"
            shift
            ;;
        -clean)  # '-clean' 인자를 확인하는 케이스 추가
            clean_mode=true
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
    shift
done

# '-clean' 인자가 주어진 경우에만 'test_clean_folder_gpu.sh' 스크립트 실행
if [ "$clean_mode" = true ]; then
    ./test_clean_folder_gpu.sh -model "${model}"
fi

# model 값에 따른 layer_num 값 설정
if [ "$model" == "densenet201" ]; then
    data_file="imagenet1k"
    layer_start=0
    layer_end=306
    layer_num=306
elif [ "$model" == "resnet152" ]; then
    data_file="imagenet1k"
    layer_start=0
    layer_end=206
    layer_num=206
elif [ "$model" == "enetb0" ]; then
    data_file="imagenet1k"
    layer_start=0
    layer_end=136
    layer_num=136
elif [ "$model" == "csmobilenet-v2" ]; then
    data_file="imagenet1k"
    layer_start=0
    layer_end=81
    layer_num=81
elif [ "$model" == "squeezenet" ]; then
    data_file="imagenet1k"
    layer_start=0
    layer_end=50
    layer_num=50
elif [ "$model" == "yolov7" ]; then
    data_file="coco"
    layer_start=0
    layer_end=143
    layer_num=143
elif [ "$model" == "yolov7-tiny" ]; then
    data_file="coco"
    layer_start=0
    layer_end=99
    layer_num=99
elif [ "$model" == "yolov4" ]; then
    data_file="coco"
    layer_start=0
    layer_end=162
    layer_num=162
elif [ "$model" == "yolov4-tiny" ]; then
    data_file="coco"
    layer_start=0
    layer_end=38
    layer_num=38
elif [ -z "$model" ]; then
    echo "Model not specified. Use -model to specify the model."
    exit 1
else
    echo "Unknown model: $model"
    exit 1
fi

# GPU-accelerated with optimal_core
# for glayer in $(seq $layer_start $layer_num); do
#     for ((rlayer = glayer + 1; rlayer < $layer_num; rlayer++)); do
#         ./darknet detector cpu-reclaiming ./cfg/${data_file}.data ./cfg/${model}.cfg ./weights/${model}.weights data/dog.jpg -num_thread 11 -glayer $glayer -rlayer $rlayer -num_exp 30
#     done
# done


# 초기 optimal_core 값을 설정
optimal_core="NULL"
gpu_infer=0.0
recaliming_infer=0.0

# GPU-accelerated & CPU-reclaiming with optimal_core
for glayer in $(seq $layer_start $layer_end); do
    sleep 1s
    echo "glayer: $glayer"
    ./darknet detector cpu-reclaiming ./cfg/${data_file}.data ./cfg/${model}.cfg ./weights/${model}.weights data/dog.jpg -num_thread 11 -glayer $glayer -num_exp 30
    sleep 1s
done
