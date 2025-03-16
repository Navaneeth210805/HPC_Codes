#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define N 10000
#define BLOCK_SIZE 256
__global__ void matrix_add(double *a, double *b, double *c, size_t pitch, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n) {
        double *rowA = (double *)((char *)a + row * pitch);
        double *rowB = (double *)((char *)b + row * pitch);
        double *rowC = (double *)((char *)c + row * pitch);

        for (int col = 0; col < n; col++) {
            rowC[col] = rowA[col] + rowB[col];
        }
    }
}

int main() {
    size_t pitch;
    

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
            h_a[i][j] = (double)(rand() % 1000 + 1);
            h_b[i][j] = (double)(rand() % 1000 + 1);
            h_c[i][j] = 0.0;
        }
    }

    double *d_a, *d_b, *d_c;
    cudaMallocPitch((void **)&d_a, &pitch, N * sizeof(double), N);
    cudaMallocPitch((void **)&d_b, &pitch, N * sizeof(double), N);
    cudaMallocPitch((void **)&d_c, &pitch, N * sizeof(double), N);

    for (int i = 0; i < N; i++) {
        cudaMemcpy2D((char *)d_a + i * pitch, pitch, h_a[i], N * sizeof(double), N * sizeof(double), 1, cudaMemcpyHostToDevice);
        cudaMemcpy2D((char *)d_b + i * pitch, pitch, h_b[i], N * sizeof(double), N * sizeof(double), 1, cudaMemcpyHostToDevice);
    }

    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    double start_time = clock();

    matrix_add<<<numBlocks, BLOCK_SIZE>>>(d_a, d_b, d_c, pitch, N);
    cudaDeviceSynchronize();

    double end_time = clock();
    printf("Parallel Matrix Addition Time: %f seconds\n", (end_time - start_time) / CLOCKS_PER_SEC);

    for (int i = 0; i < N; i++) {
        cudaMemcpy2D(h_c[i], N * sizeof(double), (char *)d_c + i * pitch, pitch, N * sizeof(double), 1, cudaMemcpyDeviceToHost);
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
