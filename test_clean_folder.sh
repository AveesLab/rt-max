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
folder_path="measure/gpu-accel_gpu/${model}/"

# 폴더 존재 여부 확인
if [ ! -d "$folder_path" ]; then
    echo "Error: Directory '$folder_path' does not exist."
    exit 1
fi

# 폴더 내의 모든 파일 삭제
# 이 명령은 폴더 내의 모든 파일과 서브디렉토리를 삭제합니다.
# 서브디렉토리 내의 파일들도 모두 삭제됩니다.
rm -rf "$folder_path"*

echo "All files in '$folder_path' have been deleted."
echo "========================================================================="