#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define ROWS 10000
#define COLS 10000

int main() {
    printf("Hello");
    int THREADS[] = {1,2,4,6,8,10,12,16,20,32,64};
    FILE *ans = fopen("NXN_Matrix_Addition.txt", "a");  
    FILE *fptr = fopen("matrix1.txt", "r");
    FILE *fptr2 = fopen("matrix1.txt","r");
    
    int i, j, k;
    
    double **matrix1 = (double **)malloc(ROWS * sizeof(double *));
    double **matrix2 = (double **)malloc(ROWS * sizeof(double *));
    double **result = (double **)malloc(ROWS * sizeof(double *));
    
    for (i = 0; i < ROWS; i++) {
        matrix1[i] = (double *)malloc(COLS * sizeof(double));
        matrix2[i] = (double *)malloc(COLS * sizeof(double));
        result[i] = (double *)malloc(COLS * sizeof(double));
        
        for (j = 0; j < COLS; j++) {
            fscanf(fptr, "%lf", &matrix1[i][j]);
            fscanf(fptr2, "%lf", &matrix2[i][j]);
        }
        // printf("%d\n",i);
    }
    
    for (i = 0; i < 1; i++) {
        omp_set_num_threads(THREADS[i]);
        double startTime = omp_get_wtime();
        
        #pragma omp parallel for collapse(2) private(j, k)
        for (j = 0; j < ROWS; j++) {
            for (k = 0; k < COLS; k++) {
                result[j][k] = matrix1[j][k] + matrix2[j][k];
            }
        }
        
        double endTime = omp_get_wtime();
        printf("Time taken by OpenMP: %f seconds for threads %d\n", endTime - startTime, THREADS[i]);
        fprintf(ans, "%f\n", endTime - startTime);
    }


    for (i = 0; i < 10; i++) {
        omp_set_num_threads(THREADS[i]);
        double startTime = omp_get_wtime();
        
        #pragma omp parallel for collapse(2) private(j, k)
        for (j = 0; j < ROWS; j++) {
            for (k = 0; k < COLS; k++) {
                result[j][k] = matrix1[j][k] + matrix2[j][k];
            }
        }
        
        double endTime = omp_get_wtime();
        printf("Time taken by OpenMP: %f seconds for threads %d\n", endTime - startTime, THREADS[i]);
        fprintf(ans, "%f\n", endTime - startTime);
    }
    
    fclose(ans);
    fclose(fptr);
    fclose(fptr2);
    
    // // Free allocated memory
    for (i = 0; i < ROWS; i++) {
        free(matrix1[i]);
        free(matrix2[i]);
        free(result[i]);
    }
    free(matrix1);
    free(matrix2);
    free(result);
    
    return 0;
}
