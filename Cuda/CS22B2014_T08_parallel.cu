#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#define N (1000*1000)

__global__ void sum_of_n(double *d_arr, double *d_sum) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ double shared_data[1024];
    
    if (tid < N) {
        shared_data[threadIdx.x] = d_arr[tid];
    }
    __syncthreads();
    
    if (threadIdx.x == 0) {
        double local_sum = 0.0;
        for (int i = 0; i < blockDim.x && tid + i < N; i++) {
            local_sum += shared_data[i];
        }
        atomicAdd(d_sum, local_sum);
    }
}

int main() {
    FILE *fptr = fopen("vector1.txt", "r");

    double *arr = (double *)malloc(N * sizeof(double));
    double sum = 0.0;
    for (int i = 0; i < N; i++) {
        fscanf(fptr, "%lf", &arr[i]);
    }
    fclose(fptr);

    double *d_arr, *d_sum;
    cudaMalloc((void **)&d_arr, N * sizeof(double));
    cudaMalloc((void **)&d_sum, sizeof(double));
    cudaMemcpy(d_arr, arr, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(d_sum, 0, sizeof(double));
    
    int threadsPerBlock = 1024;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    clock_t start = clock();
    sum_of_n<<<blocksPerGrid, threadsPerBlock>>>(d_arr, d_sum);
    cudaDeviceSynchronize();
    clock_t end = clock();
    
    cudaMemcpy(&sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost);
    
    cudaFree(d_arr);
    cudaFree(d_sum);
    free(arr);
    
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Parallel Sum: %f\n", sum);
    
    printf("Parallel execution time: %.6f seconds\n", time_taken);

    return 0;
}

