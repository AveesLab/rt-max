// gcc -o matrix_multiply matrix_multiply.c -I/usr/local/include -L/usr/local/lib -lblis -lm
// ./matrix_multiply

#include <stdio.h>
#include "blis/blis.h"

int main() {
    // 행렬의 크기를 설정
    dim_t m = 4; // C의 행 수
    dim_t n = 4; // C의 열 수
    dim_t k = 4; // A의 열 수 및 B의 행 수

    // 행렬 A, B, C를 선언 (초기화 제거)
    float a[m*k];
    float b[k*n];
    float c[m*n];

    // 스칼라 값
    float alpha = 1.0;
    float beta = 0.0;

    // A와 B의 값 설정
    float initial_a[] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };
    float initial_b[] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };
    for (int i = 0; i < m*k; ++i) {
        a[i] = initial_a[i];
    }
    for (int i = 0; i < k*n; ++i) {
        b[i] = initial_b[i];
    }

    // bli_sgemm 함수 호출
    bli_sgemm(
        BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE,
        m, n, k,
        &alpha,
        a, 1, m,   // A의 데이터, 행 간격, 열 간격
        b, 1, k,   // B의 데이터, 행 간격, 열 간격
        &beta,
        c, 1, m    // C의 데이터, 행 간격, 열 간격
    );

    // 결과 출력
    printf("Result Matrix C:\n");
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%.1f ", c[i*m+j]);
        }
        printf("\n");
    }

    return 0;
}

