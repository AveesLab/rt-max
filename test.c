1. Title
2. Background and Motivation
3. Research Objectives
4. System Model 
5. Task-parallel vs. Data-parallel
6. Data-parallel
7. Selective Accleration: Acceleration gain과 Blocking Loss Net Gain
System: M CPU cores {C 1 ​ ,C 2 ​ ,…,C M} + single shared GPU (G)
DNN Task: N layers (l i) with CPU/GPU execution time (ci, gi)
Layer-to-Resource Mapping: 𝜙 𝑖 = { 0 (CPU) , 1 (GPU) }
Total execution time: einfer​(ϕ)=i=1∑N​[ci​⋅(1−ϕi​)+gi​⋅ϕi​]
8. Selective Acceleration: 얼마나 많이 가속? 하냐에 따라 Acceleration gain과 Blocking Loss 차이
9. Selective Acceleration: 누구를 가속? 하냐에 따라 Acceleration gain과 Blocking Loss 차이
10. Blocking Loss Analysis: WCRT and BCRT
11. 단계별 전략(Stepwise 전략): DNN 레이어 수(N)가 많아질수록 가능한 CPU/GPU 배치의 조합 수는 지수적으로 증가하여(2N2^N2N) 완전 탐색(Exhaustive search)은 현실적으로 불가능함.
따라서 현실적이고 효율적인 탐색을 위해 단계별 전략(Stepwise Selective Acceleration Heuristics) 을 통해 최적의 배치를 빠르게 찾아야 함.
12. Full-freedom 전략 (2^N combination):
모든 가능한 배치를 탐색하는 방식.
가장 유연하지만 현실적으로 계산 비용이 너무 높으며, Full-freedom 전략은 레이어 간 GPU/CPU 전환이 빈번히 발생하여 GPU Idle time과 blocking time이 빈번하게 생깁니다. 이로 인해 실제 Blocking Loss가 증가하게 됩니다. GPU 가속이 과도하게 파편화(fragmented) 되어 Blocking Loss 증가로 효율이 낮음.
동일한 GPU 가속량(Acceleration Gain)을 가지는 상황에서도 어떤 레이어를 GPU에 매핑하느냐에 따라 Blocking Loss가 크게 달라지므로, 이를 명확히 정량화하여 분석하는 것이 필수적임. (GPU 가속 파편화는 일반적으로 불리함) --> 타이밍 다이어그램으로 비교해서 Blocking Loss가 커지는 것을 보여줌 (Multi-GPU Segment, Single-GPU Segment 차이 보여주면서 Single 전략인 Targeted Acceleration으로 자연스럽게 넘어감)
13. Targeted Acceleration [Start, k] (N(N+1)/2 combination):
임의의 레이어에서 시작하여 연속된 레이어를 GPU로 가속화하는 방식. 유연하면서도 탐색 공간을 현저히 줄여 효율적임.
Layer 별 다양한 Computation 특성을 가진 DNN에서 좋은 성능을 기대할 수 있음.
Targeted Acceleration은 상당히 효율적이지만, 대부분의 모델에서 실제 연산 부하가 초반부 레이어에 집중되어 있으므로 이를 고려한 Front-first 전략이 더욱 현실적입니다. 연산량이 특정 영역(특히 초반부)에 몰린 모델의 경우 반드시 최적이 아닐 수 있음.
14. Front-first 전략 [0, k] (N combination):
연산량이 많은 초반부 레이어를 우선 GPU 가속화 하는 방식.
일반적으로 DNN 모델들은 초반 레이어가 가장 연산이 집중되어 있으므로 현실적이고 효율적인 전략임.
실험적으로 DenseNet과 같은 모델에서는 가장 성능이 뛰어난 전략으로 입증됨.
15. Step-wise 전략 Summary:
사진 참고
16. Experimental Setup
17. Evaluation: Full-Freedom Acceleration
18. Evaluation: Targeted Acceleration
19. Evaluation: Front-First Acceleration
20. Conclusions and Future Work