#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "v0.h"
#include "utils.h"

knnresult kNN(double *x, double *y, uint32_t n, uint32_t m, uint32_t d, uint32_t k) {

    knnresult ret;
    MALLOC(double, ret.ndist, m*MIN(k, n));
    MALLOC(uint32_t, ret.nidx, m*MIN(k, n));
    ret.k = k; // included the query to be tested?
    ret.m = m;

    struct timespec tic;
    struct timespec toc;
    
    //TIC()
    double *distance = euclidean_distance(x, y, n, d, m); // m x n
    //TOC("Time elapsed calculating distance matrix (seconds): %lf\n")

    /* printf("\nThe distance matrix m x n: \n"); */
    /* print_dataset_yav(distance, m, n); */


    int isFirst = 1; // indeces 1 or 0 based?
    //printf("Sorting distance matrix until %uth element...\n", k);

    //TIC()

    // k-select per row
    for(uint32_t i = 0; i < m; i++) {
        double *distance_per_row; 
        distance_per_row = distance + i*n;

        uint32_t *indeces_per_row;
        MALLOC(uint32_t, indeces_per_row, n);
        for(uint32_t j = 0; j < n; j ++) indeces_per_row[j] = j + isFirst;

        if(n>k) {
            qselect_and_indeces(distance_per_row, indeces_per_row, n, k-1);
            memcpy(ret.ndist + i*k, distance_per_row, sizeof(double) * k);
            memcpy(ret.nidx + i*k, indeces_per_row, sizeof(uint32_t) * k);
        }
        else {  
            memcpy(ret.ndist + i*n, distance_per_row, sizeof(double) * n);
            memcpy(ret.nidx + i*n, indeces_per_row, sizeof(uint32_t) * n);
        }

        free(indeces_per_row);
    }

    free(distance);

    //TOC("Time elapsed calculating kNN given distance matrix (seconds): %lf\n")

    /* if (k>n) k = n; */
    /* printf("\nDistance of kNN\n"); */
    /* print_dataset_yav(ret.ndist, m, k); */
    /* printf("\nIndeces of kNN\n"); */
    /* print_indeces(ret.nidx, m, k); */

    return ret;
}
