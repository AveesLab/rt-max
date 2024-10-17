# formatted_glayer=$(printf "%03d" $glayer)


# 파일 경로 변수 설정
#  measure/gpu-accel-nano/densenet201-multithread/1thread/gpu-accel-000glayer.csv
ARCH=gpu-accel-nano
MODEL=densenet201
thread=1
glayer=0
formatted_glayer=$(printf "%03d" $glayer)
file_path="/home/avees/rt-max/measure/${ARCH}/${MODEL}-multithread/${thread}thread/gpu-accel-${formatted_glayer}glayer.csv"
echo $file_path
# 파일이 있는지 확인하는 조건문
if ls $file_path 1> /dev/null 2>&1; then
    echo "파일이 존재합니다."
else
    echo "파일이 존재하지 않습니다."
fi
