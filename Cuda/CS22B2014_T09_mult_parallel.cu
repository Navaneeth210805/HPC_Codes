#include<stdio.h>
#include<stdlib.h>
#include <cuda_runtime.h>
#include<time.h>
#define N (1000*1000)



__global__ void addn(double *d_arr,double *d_arr2,double *d_result)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        d_result[tid] = d_arr[tid] * d_arr2[tid];
    }    
}

int main()
{
    FILE *fptr = fopen("vector1.txt", "r");
    FILE *fptr2 = fopen("vector2.txt", "r");

    double *arr = (double *)malloc(N * sizeof(double));
    double *arr2 = (double *)malloc(N * sizeof(double));
    double *result = (double *)malloc(N * sizeof(double));

    for (int i = 0; i < N; i++) {
        fscanf(fptr, "%lf", &arr[i]);
        fscanf(fptr2, "%lf", &arr2[i]);
    }
    fclose(fptr);
    fclose(fptr2);

    double *d_arr, *d_arr2, *d_result;
    cudaMalloc((void **)&d_arr, N * sizeof(double));
    cudaMalloc((void **)&d_arr2, N * sizeof(double));
    cudaMalloc((void **)&d_result, N * sizeof(double));
    cudaMemcpy(d_arr, arr, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_arr2, arr2, N * sizeof(double), cudaMemcpyHostToDevice);
    int threads = 1024;
    double start = clock();
    addn<<<(N+threads-1 )/threads,threads>>>(d_arr, d_arr2, d_result);
    cudaDeviceSynchronize();
    double end = clock();
    cudaMemcpy(result, d_result, N * sizeof(double), cudaMemcpyDeviceToHost);
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Parallel Time taken: %f seconds\n", time_taken);

    free(arr);
    free(arr2);
    free(result);
    cudaFree(d_arr);
    cudaFree(d_arr2);
    cudaFree(d_result);

    return 0;
}
