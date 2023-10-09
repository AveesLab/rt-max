import argparse
import csv
import os
import math
import numpy as np

def gather_e_infer(directory, output_filename):
    # 초기화
    result_prerocess= ['e_prerocess']
    result_infer = ['e_infer']
    result_postprocess = ['e_postprocess']

    # 1부터 11까지 돌면서 해당 파일명의 데이터를 가져오기
    for i in range(1, 12): 
        file_name = f'sequential_{i:02}blas.csv' 
        file_path = os.path.join(directory, file_name)

        # 파일 존재 여부 확인
        if not os.path.exists(file_path):
            print(f"Error: {file_path} does not exist in the directory.")
            continue

        # 파일의 데이터를 읽어와서 평균값 계산
        with open(file_path, 'r') as f:
            reader = csv.DictReader(f)
            e_prerocess_vals = []
            e_infer_vals = []
            e_postprocess_vals = []

            for row in reader:
                e_prerocess_vals.append(float(row['e_preprocess']))
                e_infer_vals.append(float(row['e_infer']))
                e_postprocess_vals.append(float(row['e_postprocess']))

            result_prerocess.append(sum(e_prerocess_vals) / len(e_prerocess_vals))
            result_infer.append(sum(e_infer_vals) / len(e_infer_vals))
            result_postprocess.append(sum(e_postprocess_vals) / len(e_postprocess_vals))

    # 평균값 데이터를 csv 파일로 작성
    with open(output_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(zip(result_prerocess, result_infer, result_postprocess))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gather e_gpu_infer from given model")
    parser.add_argument("-model", type=str, required=True, help="Model name")
    args = parser.parse_args()

    directory = f"/home/avees/baseline/darknet/measure/sequential-multiblas/{args.model}/"
    output_filename = f"./measure/layer_time/{args.model}/sequential_inference_list.csv"

    gather_e_infer(directory, output_filename)
