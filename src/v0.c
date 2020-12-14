#include <stdio.h>
#include "include/main.h"
#include <cblas.h>
#include <string.h>


knnresult kNN(double *x, double *y, uint32_t n, uint32_t m, uint32_t d, uint32_t k) {

    knnresult ret;
    ret.ndist = (double*)malloc(m * k * sizeof(double));
    ret.nidx  = (uint32_t*)malloc(m * k * sizeof(uint32_t));
    ret.k = 0;
    ret.m = 0;

    double distance[m][n];
    //memset(distance, 0.0, m * n * sizeof(double));

    //  D = sqrt(sum(X.^2,2) - 2 * X*Y.' + sum(Y.^2,2).');

    return ret;
}
