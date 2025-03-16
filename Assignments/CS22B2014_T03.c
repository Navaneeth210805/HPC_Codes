#include<stdio.h>
#include<stdlib.h>
#include<omp.h>

#define MAX_SIZE 1000000
int main()
{
    FILE *fptr1 = fopen("vector1.txt","r");
    FILE *fptr2 = fopen("vector2.txt","r");

    double * vector1 = (double *)malloc(MAX_SIZE * sizeof(double));
    double * vector2 = (double *)malloc(MAX_SIZE * sizeof(double));
    double * add = (double *)malloc(MAX_SIZE * sizeof(double));
    double * mul = (double *)malloc(MAX_SIZE * sizeof(double));

    unsigned count = 0;
    while (count < MAX_SIZE && fscanf(fptr1, "%lf", &vector1[count]) == 1) {
        count++;
    }
    fclose(fptr1);

    count = 0;
    while (count < MAX_SIZE && fscanf(fptr2, "%lf", &vector2[count]) == 1) {
        count++;
    }
    fclose(fptr2);

    int i;
    double startTime = omp_get_wtime();
    #pragma omp parallel shared(vector1,vector2,add) private(i)
    {
        i = 0;
        #pragma omp for
        for(int i =0 ;i<MAX_SIZE;i++)
        {
            add[i] = vector1[i] + vector2[i];
        }
    }
    double endTime = omp_get_wtime();
    printf("Adder Time Taken: %f\n", endTime -startTime);

    startTime = omp_get_wtime();
    #pragma omp parallel shared(vector1,vector2,mul) private(i)
    {
        i = 0;
        #pragma omp for
        for(int i =0 ;i<MAX_SIZE;i++)
        {
            mul[i] = vector1[i] * vector2[i];
        }
    }
    endTime = omp_get_wtime();
    printf("Multiplier Time Taken: %f\n", endTime -startTime);
    free(vector1);
    free(vector2);
    free(add);
    free(mul);
    return 0;
}