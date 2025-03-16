#include<stdio.h>
#include<stdlib.h>
#include<time.h>

int main()
{
    int N = 10000;
    double **a1 = (double **)malloc(N * sizeof(double));
    double **a2 = (double **)malloc(N *sizeof(double));
    for(int i =0 ;i<N;i++)
    {
        a1[i] = (double *)malloc(N * sizeof(double));
        a2[i] = (double *)malloc(N *sizeof(double));
        for(int j =0 ;j<N;j++)
        {
            a1[i][j] = (double)(rand() %1000 + 1);
            a2[i][j] = (double)(rand() %1000 + 1);
        }
    }


    double **a3 = (double **)malloc(N * sizeof(double));
    double start_time = clock();
    for(int i =0 ;i<N;i++)
    {
        a3[i] = (double *)malloc(N * sizeof(double));
        for(int j = 0;j<N;j++)
        {
            a3[i][j] = a1[i][j] + a2[i][j];
        }
    }
    double end_time = clock();
    printf("Serial Matrix Addition Time Taken %f",(end_time - start_time)/ CLOCKS_PER_SEC);

    free(a1);
    free(a2);
    free(a3);
    return 0;
}