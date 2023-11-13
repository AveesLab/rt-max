#!/bin/bash

# 추론 스크립트 파일 경로 설정
INFER_SCRIPT="sequential_test.sh -model densenet201"

# CSV 로그 파일 경로 설정
CSV_LOG="cpu_energy_log.csv"

# CSV 파일 헤더 작성
echo "Time, CPU Energy" > "$CSV_LOG"

# tegrastats를 사용하여 에너지 사용량 모니터링 (초당)
(sudo tegrastats --interval 1000 | while read line; do
    # 현재 시간과 CPU 에너지 사용량 추출
    TIMESTAMP=$(date +"%T.%3N")
    CPU_ENERGY=$(echo $line | grep -oP 'CPU \K\d+')

    # CSV 파일에 기록
    echo "$TIMESTAMP, $CPU_ENERGY" >> "$CSV_LOG"
done) &

# tegrastats 프로세스 ID 저장
TEGRASTATS_PID=$!

# DNN 추론 실행
bash "$INFER_SCRIPT"

# 추론이 끝나면 tegrastats 종료
kill $TEGRASTATS_PID

echo "DNN 추론 완료. 에너지 사용량은 $CSV_LOG 파일에 기록되었습니다."
