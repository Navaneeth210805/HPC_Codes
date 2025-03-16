#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define N 1000
#define BLOCK_SIZE 256

__global__ void matrix_multiply(double *a, double *b, double *c, size_t pitchA, size_t pitchB, size_t pitchC, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n) {
        double *rowA = (double *)((char *)a + row * pitchA);
        double *rowC = (double *)((char *)c + row * pitchC);
        for (int col = 0; col < n; col++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                double *rowB = (double *)((char *)b + k * pitchB);
                sum += rowA[k] * rowB[col];
            }
            rowC[col] = sum;
        }
    }
}

int main() {
    size_t pitchA, pitchB, pitchC;
    double **h_a = (double **)malloc(N * sizeof(double *));
    double **h_b = (double **)malloc(N * sizeof(double *));
    double **h_c = (double **)malloc(N * sizeof(double *));
    
    for (int i = 0; i < N; i++) {
        h_a[i] = (double *)malloc(N * sizeof(double));
        h_b[i] = (double *)malloc(N * sizeof(double));
        h_c[i] = (double *)malloc(N * sizeof(double));
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            h_a[i][j] = (double)(rand() % 10 + 1);
            h_b[i][j] = (double)(rand() % 10 + 1);
            h_c[i][j] = 0.0;
        }
    }

    double *d_a, *d_b, *d_c;
    cudaMallocPitch((void **)&d_a, &pitchA, N * sizeof(double), N);
    cudaMallocPitch((void **)&d_b, &pitchB, N * sizeof(double), N);
    cudaMallocPitch((void **)&d_c, &pitchC, N * sizeof(double), N);

    for (int i = 0; i < N; i++) {
        cudaMemcpy2D((char *)d_a + i * pitchA, pitchA, h_a[i], N * sizeof(double), N * sizeof(double), 1, cudaMemcpyHostToDevice);
        cudaMemcpy2D((char *)d_b + i * pitchB, pitchB, h_b[i], N * sizeof(double), N * sizeof(double), 1, cudaMemcpyHostToDevice);
    }

    double start_time = clock();
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    matrix_multiply<<<numBlocks, BLOCK_SIZE>>>(d_a, d_b, d_c, pitchA, pitchB, pitchC, N);
    cudaDeviceSynchronize();
    double end_time = clock();
    printf("CUDA Matrix Multiplication Time: %f seconds\n", (end_time - start_time) / CLOCKS_PER_SEC);

    for (int i = 0; i < N; i++) {
        cudaMemcpy2D(h_c[i], N * sizeof(double), (char *)d_c + i * pitchC, pitchC, N * sizeof(double), 1, cudaMemcpyDeviceToHost);
    }

    for (int i = 0; i < N; i++) {
        free(h_a[i]);
        free(h_b[i]);
        free(h_c[i]);
    }
    free(h_a);
    free(h_b);
    free(h_c);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
