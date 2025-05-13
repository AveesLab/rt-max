from itertools import combinations
import pandas as pd
import sys
import argparse
import os

def generate_all_contiguous_subsets(n):
    """n까지의 범위에서 모든 연속된 부분집합 생성"""
    subsets = []
    for i in range(n):
        for j in range(i + 1, n + 1):
            subsets.append(tuple(range(i, j)))
    return subsets

def generate_all_contiguous_partitions(n):
    """0부터 n-1까지의 숫자에 대한 모든 연속 분할 생성"""
    results = []
    def backtrack(start, path):
        if start == n:
            results.append(path[:])
            return
        for end in range(start + 1, n + 1):
            path.append(tuple(range(start, end)))
            backtrack(end, path)
            path.pop()
    backtrack(0, [])
    return results

def greedy_set_cover(universe, subsets):
    """탐욕적 집합 커버 알고리즘"""
    covered = set()
    cover = []
    # 집합을 복사하여 원본을 수정하지 않도록 함
    subsets_copy = subsets.copy()
    
    while covered != universe and subsets_copy:
        best_subset = max(subsets_copy, key=lambda s: len(set(s) - covered))
        if not set(best_subset) - covered:  # 더 이상 새로운 원소를 커버하지 않으면 중단
            break
        cover.append(best_subset)
        covered.update(best_subset)
        subsets_copy.remove(best_subset)
    
    return cover

def main():
    # 명령줄 인자 파싱
    parser = argparse.ArgumentParser(description='연속 부분집합을 커버하는 최소 분할 집합 찾기')
    parser.add_argument('-n', type=int, default=10, help='숫자 범위 크기 (기본값: 10)')
    parser.add_argument('-model', type=str, required=True, help='모델 이름 (필수)')
    args = parser.parse_args()
    
    n = args.n
    model = args.model
    
    # 모델별 디렉토리 및 파일 경로 생성
    model_dir = f'measure/gpu_segments/{model}'
    os.makedirs(model_dir, exist_ok=True)
    
    output_file = f'{model_dir}/gpu_segment_partitions_{model}.csv'
    counts_file = f'{model_dir}/gpu_segment_coverage_{model}.csv'
    
    print(f"실행 중... n = {n}, model = {model}")
    print(f"출력 디렉토리: {model_dir}")
    
    # 모든 연속 부분집합 생성
    all_subsets = generate_all_contiguous_subsets(n)
    universe = set(all_subsets)
    
    print(f"생성된 부분집합 개수: {len(universe)} (기대값: {n*(n+1)//2})")
    
    # 모든 연속 분할 생성
    all_partitions = generate_all_contiguous_partitions(n)
    
    print(f"생성된 분할 개수: {len(all_partitions)}")
    
    # 각 분할이 커버하는 부분집합 매핑
    partition_coverage = []
    for partition in all_partitions:
        partition_coverage.append(set(partition))
    
    # 탐욕적 집합 커버 실행
    print("탐욕적 집합 커버 알고리즘 실행 중...")
    selected_partitions = greedy_set_cover(universe, partition_coverage)
    
    # 검증 및 누락된 서브셋 찾기
    covered_subsets = set()
    for partition in selected_partitions:
        covered_subsets.update(partition)
    
    uncovered = universe - covered_subsets
    
    if uncovered:
        print(f"⚠️ 누락된 서브셋이 있습니다: {len(uncovered)}개")
        print("다음 서브셋들이 커버되지 않았습니다:")
        for s in sorted(uncovered):
            print(s)
        print("\n⚠️ 조건 불만족: 일부 서브셋이 커버되지 않았습니다!")
    else:
        total_subsets = n * (n + 1) // 2
        print(f"✅ 모든 {total_subsets}개의 연속된 서브셋이 완전히 커버되었습니다!")
    
    # 선택된 분할 출력
    print(f"\n선택된 분할 개수: {len(selected_partitions)}")
    for i, partition in enumerate(selected_partitions):
        print(f"분할 {i+1}: {list(partition)}")
    
    # 결과 저장
    df = pd.DataFrame({
        "Partition_Index": list(range(len(selected_partitions))),
        "Partition": [list(p) for p in selected_partitions]
    })
    
    df.to_csv(output_file, index=False)
    print(f"✔️ 저장 완료: {output_file}")
    
    # --- 서브셋 등장 횟수 세기 ---
    subset_counts = {s: 0 for s in all_subsets}
    for partition in selected_partitions:
        for subset in partition:
            subset_counts[subset] += 1
    
    # 각 서브셋이 최소 1회 이상 등장하는지 확인
    zero_appearance_subsets = [s for s, count in subset_counts.items() if count == 0]
    
    if zero_appearance_subsets:
        print(f"\n⚠️ 조건 불만족: {len(zero_appearance_subsets)}개의 서브셋이 한 번도 등장하지 않았습니다!")
        print("다음 서브셋들이 한 번도 등장하지 않았습니다:")
        for s in sorted(zero_appearance_subsets):
            print(s)
    else:
        print("\n✅ 조건 만족: 모든 서브셋이 최소 1회 이상 등장합니다!")
    
    # --- 결과 저장 ---
    count_df = pd.DataFrame([
        {"Subset": s, "Count": c}
        for s, c in sorted(subset_counts.items())
    ])
    
    count_df.to_csv(counts_file, index=False)
    print(f"✔️ 저장 완료: {counts_file}")
    
    # 메타데이터 파일 저장 (추가 정보)
    metadata = {
        "Model": [model],
        "Layers": [n],
        "Total_Subsets": [len(universe)],
        "Selected_Partitions": [len(selected_partitions)]
    }
    
    metadata_df = pd.DataFrame(metadata)
    metadata_file = f'{model_dir}/metadata_{model}.csv'
    metadata_df.to_csv(metadata_file, index=False)
    print(f"✔️ 메타데이터 저장 완료: {metadata_file}")

if __name__ == "__main__":
    main()