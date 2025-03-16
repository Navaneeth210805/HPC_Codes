#include <stdio.h>
#include <stdlib.h>
#include<omp.h>

#define MAX_NUMBERS 1000000

int main() {
    FILE *file = fopen("large_doubles.txt", "r");
    double *numbers = (double *)malloc(MAX_NUMBERS * sizeof(double));
    unsigned int count = 0;
    while (count < MAX_NUMBERS && fscanf(file, "%lf", &numbers[count]) == 1) {
        count++;
    }
    fclose(file);


// Using Reduction Construct
    
    double startTime = omp_get_wtime();
    double sum = 0;
    unsigned int i = 0;
    #pragma omp parallel private (i) shared(numbers) reduction(+:sum)
    {
        #pragma omp for
        
        for(int i = 0;i<MAX_NUMBERS;i++)
            {
                sum += numbers[i];
            }
    }
    double endTime = omp_get_wtime();
    printf("Reduction Time taken: %f\n", endTime - startTime);
    // printf("Sum %f\n",sum);


// Using Critical Section

    startTime = omp_get_wtime();
    sum = 0;
    i = 0;
    double psum;
    #pragma omp parallel private (i,psum) shared(numbers,sum)
    {
        psum = 0;
        #pragma omp for
        
        for(int i = 0;i<MAX_NUMBERS;i++)
            {
                psum += numbers[i];
            }
        
        #pragma omp critical(dosum)
        {
            sum += psum;
        }
    }
    endTime = omp_get_wtime();
    printf("Critical Time taken: %f\n", endTime - startTime);
    // printf("Sum %f\n",sum);

    free(numbers);
    return 0;
}
