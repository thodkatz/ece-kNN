#include <stdio.h>
#include <stdlib.h>
#include "include/main.h"
#include <cblas.h>
#include <string.h>
#include <math.h>


extern void print_dataset(double *x, uint32_t n, uint32_t d);

knnresult kNN(double *x, double *y, uint32_t n, uint32_t m, uint32_t d, uint32_t k) {

    knnresult ret;
    ret.ndist = (double*)malloc(m * k * sizeof(double));
    ret.nidx  = (uint32_t*)malloc(m * k * sizeof(uint32_t));
    ret.k = 0;
    ret.m = 0;

    //  D = sqrt(sum(X.^2,2) - 2 * X*Y.' + sum(Y.^2,2).');
    
    // d1 = sum(X.^2,2)
    double d1[n];
    for (uint32_t i = 0; i < n; i++) {
        d1[i] = 0;
        for (uint32_t j = 0; j < d; j++) {
            double temp = x[i*d + j] * x[i*d + j];
            printf("%lf ", temp);
            d1[i] += x[i*d + j] * x[i*d + j]; // should I use dot product?
        }
        printf("\n");
        printf("%lf\n", d1[i]);
    }

    for (uint32_t i = 0; i < n; i++) printf("%lf ", d1[i]);

    // d2 = sum(Y.^2,2)
    double d2[m];
    for (uint32_t i = 0; i < m; i++) {
        d2[i] = 0;
        for (uint32_t j = 0; j < d; j++) {
            d2[i] += y[i*d + j] * y[i*d + j]; // should I use dot product?
        }
    }

    printf("\n");
    for (uint32_t i = 0; i < m; i++) printf("%lf ", d2[i]);
    printf("\n");

    // D = -2 * X * Y.'
    double *distance = (double*)malloc(n * m * sizeof(double));
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, m, d, -2, x, d, y, d, 0, distance, m);

    print_dataset(distance, n, m);


    // D = sqrt(D + d1 + d2)
    for (uint32_t i = 0; i < n; i++) {
        for (uint32_t j = 0; j < m; j++) {
            distance[i*m + j] += d1[i] + d2[j];
            distance[i*m + j] = sqrt(distance[i*m + j]);
        }
    }

    print_dataset(distance, n, m);


    return ret;
}
