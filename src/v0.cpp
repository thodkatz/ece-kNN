#include <stdio.h>
#include <stdlib.h>
#include "include/main.h"
#include <cblas.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <algorithm> 

#define DDOT 0

#define CPP 0

extern void print_dataset(double *x, uint32_t n, uint32_t d);
extern void print_dataset_yav(double *x, uint32_t n, uint32_t d);

void transpose(double *src, double *dst, const uint32_t N, const uint32_t M);

knnresult kNN(double *x, double *y, uint32_t n, uint32_t m, uint32_t d, uint32_t k) {

    knnresult ret;
    ret.ndist = (double*)malloc(m * k * sizeof(double));
    ret.nidx  = (uint32_t*)malloc(m * k * sizeof(uint32_t));
    ret.k = k; // included the query to be tested?
    ret.m = m;

    struct timespec tic;
    struct timespec toc;
    //  D = sqrt(sum(X.^2,2) - 2 * X*Y.' + sum(Y.^2,2).');
    
    // d1 = sum(X.^2,2)
    clock_gettime(CLOCK_MONOTONIC, &tic);
    double d1[n];
#if DDOT == 0
    for (uint32_t i = 0; i < n; i++) {
        d1[i] = 0;
        for (uint32_t j = 0; j < d; j++) {
            d1[i] += x[i*d + j] * x[i*d + j]; 
        }
    }
#elif DDOT == 1
    for (uint32_t i = 0; i < n; i++) d1[i] = cblas_ddot(d, x + i*d, 1, x + i*d, 1);
#endif

    // d2 = sum(Y.^2,2)
    double d2[m];
#if DDOT == 0
    for (uint32_t i = 0; i < m; i++) {
        d2[i] = 0;
        for (uint32_t j = 0; j < d; j++) {
            d2[i] += y[i*d + j] * y[i*d + j]; 
        }
    }
#elif DDOT == 1
    for (uint32_t i = 0; i < m; i++) d2[i] = cblas_ddot(d, y + i*d, 1, y + i*d, 1);
#endif

    // D = -2 * X * Y.'
    double *distance = (double*)malloc(n * m * sizeof(double));
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, m, d, -2, x, d, y, d, 0, distance, m);

    // D = sqrt(D + d1 + d2)
    for (uint32_t i = 0; i < n; i++) {
        for (uint32_t j = 0; j < m; j++) {
            distance[i*m + j] += d1[i] + d2[j];
            distance[i*m + j] = sqrt(distance[i*m + j]);
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &toc);
    double diff = diff_time(tic, toc);
    printf("Time elapsed calculating distance matrix (seconds): %lf\n", diff);

    //print_dataset(distance, n, m);
    print_dataset_yav(distance, n, m);

    /* performance issue (cache): nxm and we need find the kth element per column for a row wise array */
    double *distance_t = (double*)malloc(m * n * sizeof(double)); // mxn
    transpose(distance, distance_t, n, m);
    printf("\nThe transpose matrix m x n: \n");
    print_dataset_yav(distance_t, m, n);

    /* we need to be compatible with elearning tester */
    printf("\nSorting distance matrix until %uth element...\n", k);
    for(uint32_t i = 0; i < m; i++) {
        double *mth_array = distance_t + i*n; 
        std::nth_element(mth_array, mth_array + k-1, mth_array + n);
        for(uint32_t j = 0; j < k; j++) ret.ndist[i*k + j] = mth_array[j];
    }

    print_dataset_yav(ret.ndist, m, k);

    return ret;
}

void transpose(double *src, double *dst, const uint32_t N, const uint32_t M) {
    //#pragma omp parallel for
    for(uint32_t n = 0; n<N*M; n++) {
        uint32_t i = n/N;
        uint32_t j = n%N;
        dst[n] = src[M*j + i];
    }
}
