#include<stdio.h>
#include<stdlib.h>
#include <cuda_runtime.h>
#include<time.h>
#define N (1000*1000)
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

    double start = clock();
    for(int i = 0;i<N;i++)
    {
        result[i] = arr[i] * arr2[i];
    }
    double end = clock();
    double time_taken1 = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Serial Time taken: %f seconds\n", time_taken1);

    return 0;
}