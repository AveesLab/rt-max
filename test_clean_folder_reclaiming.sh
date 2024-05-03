#!/bin/bash

# 기본 모델 이름 설정 (필요한 경우)
model=""

# 파라미터 처리
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -model)
            model="$2"
            shift 2  # 파라미터와 값을 모두 건너뛰기
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

# 모델 파라미터가 필요함을 확인
if [[ -z "$model" ]]; then
    echo "Error: Model parameter is required. Usage: $0 -model [model_name]"
    exit 1
fi

# 폴더 경로 설정
folder_path2="measure/cpu-reclaiming/${model}/"

# 새로운 폴더 경로의 존재 여부 확인
if [ ! -d "$folder_path2" ]; then
    echo "Error: Directory '$folder_path2' does not exist."
    exit 1
fi

# 하위 폴더 안의 파일만 삭제
for dir in "$folder_path2"*/; do
    find "$dir" -type f -exec rm -f {} +
done

echo "All files within subfolders of '$folder_path2' have been deleted."
echo "========================================================================="
