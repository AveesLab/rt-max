#!/bin/bash

# 모든 CPU 코어 번호를 가져옵니다.
num_cores=$(nproc)

# 각 코어에 대해 친화도가 설정된 스레드가 있는지 확인합니다.
for ((i=0; i<num_cores; i++)); do
    # 비어있다고 가정하고 시작합니다.
    empty=true

    # 시스템의 모든 스레드의 친화도 설정을 확인합니다.
    for pid in $(pgrep -x "bash")  # 예시로 bash 프로세스만 검사
    do
        # 특정 프로세스의 친화도를 확인합니다.
        affinity=$(taskset -cp $pid 2>/dev/null | cut -d ":" -f 2)

        # 현재 코어가 친화도 설정에 포함되어 있는지 확인합니다.
        if [[ $affinity == *"$i"* ]]; then
            empty=false
            break
        fi
    done

    # 이 코어가 비어 있는지 결과를 출력합니다.
    if $empty; then
        echo "CPU Core $i is empty."
    else
        echo "CPU Core $i is used."
    fi
done
