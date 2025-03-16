#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define N (1000*1000)

_global_ void dot_product(double *arr, double *arr2, double *result) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    _shared_ double shared_data[1024];
    
    if (tid < N) {
        shared_data[threadIdx.x] = arr[tid] * arr2[tid];
    }
    __syncthreads();
    
    if (threadIdx.x == 0) {
        double local_sum = 0.0;
        for (int i = 0; i < blockDim.x && tid + i < N; i++) {
            local_sum += shared_data[i];
        }
        atomicAdd(result, local_sum);
    }
}

int main() {
    FILE *fptr = fopen("vector1.txt", "r");
    FILE *fptr2 = fopen("vector2.txt", "r");

    double *arr = (double *)malloc(N * sizeof(double));
    double *arr2 = (double *)malloc(N * sizeof(double));
    double result = 0.0;

    for (int i = 0; i < N; i++) {
        fscanf(fptr, "%lf", &arr[i]);
        fscanf(fptr2, "%lf", &arr2[i]);
    }
    fclose(fptr);
    fclose(fptr2);

    double *d_arr, *d_arr2, *d_result;
    cudaMalloc((void **)&d_arr, N * sizeof(double));
    cudaMalloc((void **)&d_arr2, N * sizeof(double));
    cudaMalloc((void **)&d_result, sizeof(double));

    cudaMemcpy(d_arr, arr, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_arr2, arr2, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(d_result, 0, sizeof(double));

    int threads = 1024;
    int blocks = (N + threads - 1) / threads;

    clock_t start = clock();
    
    dot_product<<<blocks, threads>>>(d_arr, d_arr2, d_result);
    
    cudaDeviceSynchronize();

    clock_t end = clock();
    
    cudaMemcpy(&result, d_result, sizeof(double), cudaMemcpyDeviceToHost);

    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Parallel Time Taken: %f seconds\n", time_taken);
    printf("Dot Product: %f\n", result);

    free(arr);
    free(arr2);
    cudaFree(d_arr);
    cudaFree(d_arr2);
    cudaFree(d_result);

    return 0;
}