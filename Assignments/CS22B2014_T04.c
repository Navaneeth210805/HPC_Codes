#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define MAX_NUMBERS 1000000

int main() {
    FILE *fptr1 = fopen("vector1.txt", "r");
    FILE *fptr2 = fopen("vector2.txt", "r");

    double *a = malloc(MAX_NUMBERS * sizeof(double));
    double *b = malloc(MAX_NUMBERS * sizeof(double));
    double *mul_result = malloc(MAX_NUMBERS * sizeof(double));


    for (int i = 0; i < MAX_NUMBERS; i++) 
    {
        fscanf(fptr1, "%lf", &a[i]);
        fscanf(fptr2, "%lf", &b[i]);
    }
    fclose(fptr1);
    fclose(fptr2);

    double mul_time, sum_time, start_mul, start_sum;
    double sum = 0.0;

    start_mul = omp_get_wtime();

    #pragma omp parallel
    {
        #pragma omp for
        for (int i = 0; i < MAX_NUMBERS; i++) {
            mul_result[i] = a[i] * b[i];
        }
    }

    mul_time = omp_get_wtime() - start_mul;

    start_sum = omp_get_wtime();

    #pragma omp parallel
    {
        double psum = 0.0;
        #pragma omp for
        for (int i = 0; i < MAX_NUMBERS; i++) {
            psum += mul_result[i];
        }
        #pragma omp critical
        sum += psum;
    }

    sum_time = omp_get_wtime() - start_sum;

    printf("Multiplication Time: %f\nSum Time: %f\n", mul_time, sum_time);
    printf("Sum of products: %f\n", sum);

    free(a);
    free(b);
    free(mul_result);

    return 0;
}
