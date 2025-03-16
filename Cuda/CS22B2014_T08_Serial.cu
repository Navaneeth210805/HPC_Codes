#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#define N (1000*1000)

int main() {
    FILE *fptr = fopen("vector1.txt", "r");

    double *arr = (double *)malloc(N * sizeof(double));
    for (int i = 0; i < N; i++) {
        fscanf(fptr, "%lf", &arr[i]);
    }
    fclose(fptr);


    double serial_sum = 0.0;
    clock_t start = clock();
    for (int i = 0; i < N; i++) {
        serial_sum += arr[i];
    }
    clock_t end = clock();
    double time_taken1 = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Serial sum: %.2lf\n", serial_sum);
    printf("Serial execution time: %.6f seconds\n", time_taken1);

    return 0;
}
