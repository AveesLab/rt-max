#!/bin/bash

# 평균을 계산하고 반환하는 함수
calculate_average() {
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

# 함수 사용 예
file_path="./measure/gpu-accel_gpu/densenet201/gpu-accel_000glayer.csv"
average=$(calculate_average "$file_path" 28)

# 반환된 평균값 출력
if [ "$average" != "NaN" ]; then
    echo "Calculated average: $average"
else
    echo "No data to process."
fi