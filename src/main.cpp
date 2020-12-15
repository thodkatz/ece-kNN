#include <cblas.h>
#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <stdlib.h>
#include "include/main.h"
#include <math.h>

double diff_time (struct timespec start, struct timespec end);
void print_dataset(double *x, uint32_t n, uint32_t d);
void print_dataset_yav(double *x, uint32_t n, uint32_t d);

int main(int argc, char *argv[])
{

    uint32_t n = 10;
    uint32_t d = 4;
    uint32_t m = 8;
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

    print_dataset(x, n, d);
    printf("\n");
    print_dataset(y, m, d);
    
    printf("\n<----------Version 0---------->\n");
    knnresult ret;
    ret = kNN(x, y, n, m, d, k);
    //print_knn();

    return 0;
}

// print 2d array in matlab format (row wise)
void print_dataset(double *x, uint32_t n, uint32_t d) {
    printf("[ ");
    for (uint32_t i = 0; i < n; i++) {
        for (uint32_t j = 0; j < d; j++) {
            printf("%lf ", x[i*d + j]);
        }
        if (i != n-1) printf("; ");
        else printf("]\n");
    }
}

// print 2d array in python format (row wise)
void print_dataset_yav(double *x, uint32_t n, uint32_t d) {
    for (uint32_t i = 0; i < n; i++)
    {
        if (i == 0)
            printf("[[");
        else
            printf(" [");
        for (uint32_t j = 0; j < d; j++)
            printf("%lf,", x[i*d + j]);
        if (i == n -1)
            printf("]]\n");
        else
            printf("],\n");
    }

}
