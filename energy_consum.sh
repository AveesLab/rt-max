#!/bin/bash

output_file="cpu_energy_usage.csv"  # 출력 파일명

# CSV 파일 헤더 작성
echo "Timestamp,CPU Power Consumption (mW)" > "$output_file"

# tegrastats 실행 및 CPU 전력 소모량 추출
(sudo tegrastats --interval 1000 | while read line; do
  timestamp=$(date +%s%3N)  # 밀리초 단위 타임스탬프
  cpu_power=$(echo $line | grep -oP 'VDD_CPU_CV \K\d+mW')  # CPU 전력 소모량 추출
  
  # CSV 파일에 측정값 추가
  echo "$timestamp,$cpu_power" >> "$output_file"
done) &

# 배경에서 tegrastats를 실행하기 위해 프로세스 ID를 저장합니다.
TEGRASTATS_PID=$!

# 여기에 다른 명령이나 스크립트를 실행할 수 있습니다.
./sequential_test.sh -model densenet201

# 스크립트 실행이 완료되면 tegrastats 프로세스 종료
kill $TEGRASTATS_PID

echo "모니터링 완료. CPU 에너지 사용량은 $output_file 파일에 기록되었습니다."
