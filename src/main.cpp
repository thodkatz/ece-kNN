#include <cblas.h>
#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <stdlib.h>
#include "include/main.h"
#include <math.h>

double diff_time (struct timespec start, struct timespec end);
void print_dataset(double *array, uint32_t row, uint32_t col);
void print_dataset_yav(double *array, uint32_t row, uint32_t col);
void print_indeces(uint32_t *array, uint32_t row, uint32_t col);

int main(int argc, char *argv[])
{


    printf("\n<----------Version 0 Prerequisites---------->\n");
    uint32_t n = 10000;
    uint32_t d = 40;
    uint32_t m = 80;
    uint32_t k = 3;
    uint32_t num_procs = 2;
    if (m > n) {
        printf("Number of query elements exceeded the number of elements in the corpus set\n");
        return -1;
    }
    if (k > n) {
        printf("Number of nearest elements exceeded the number of elements in the corpus set\n");
        return -1;
    }
    if (num_procs > n) {
        printf("The number of processors exceeded the number of elements in the corpus set\n");
        num_procs = n;
    }

    //srand(time(NULL));
    srand(1);
    printf("n = %u, d = %u, m = %u, k = %u\n", n, d, m, k);
    printf("Random generated corpus...\n");
    double *x = (double*)malloc(n * d * sizeof(double));
    double *y = (double*)malloc(m * d * sizeof(double)); 
    for (uint32_t i = 0; i< n; i++) {
        for (uint32_t j = 0; j < d; j++) {
            x[i*d + j] = (double)(rand()%100);
        }
    }

    printf("Random generated query...\n");
    for (uint32_t i = 0; i< m; i++) {
        uint32_t random = rand()%n; // there are duplicate indexes
        for (uint32_t j = 0; j < d; j ++) {
            y[i*d + j] = x[random*d + j];
        }
    }

    /* print_dataset(x, n, d); */
    /* printf("\n"); */
    /* print_dataset(y, m, d); */
    
    printf("\n<----------Version 0---------->\n");
    knnresult ret;
    ret = kNN(x, y, n, m, d, k);
    /* printf("\nDistance of kNN\n"); */
    /* print_dataset_yav(ret.ndist, m, k); */
    /* printf("\nIndeces of kNN\n"); */
    /* print_indeces(ret.nidx, m, k); */

    return 0;
}

// print 2d array in matlab format (row wise)
void print_dataset(double *array, uint32_t row, uint32_t col) {
    printf("[ ");
    for (uint32_t i = 0; i < row; i++) {
        for (uint32_t j = 0; j < col; j++) {
            printf("%lf ", array[i*col + j]);
        }
        if (i != row-1) printf("; ");
        else printf("]\n");
    }
}

// print 2d array in python format (row wise)
void print_dataset_yav(double *array, uint32_t row, uint32_t col) {
    for (uint32_t i = 0; i < row; i++)
    {
        if (i == 0)
            printf("[[");
        else
            printf(" [");
        for (uint32_t j = 0; j < col; j++)
            if (j != col -1) printf("%lf,", array[i*col + j]);
            else printf("%lf", array[i*col + j]);
        if (i == row -1)
            printf("]]\n");
        else
            printf("],\n");
    }

}

void print_indeces(uint32_t *array, uint32_t row, uint32_t col) {
    for (uint32_t i = 0; i < row; i++)
    {
        if (i == 0)
            printf("[[");
        else
            printf(" [");
        for (uint32_t j = 0; j < col; j++) {
            if (j != col -1) printf("%u,", array[i*col + j]);
            else printf("%u", array[i*col + j]);
        }
        if (i == row -1)
            printf("]]\n");
        else
            printf("],\n");
    }
}
