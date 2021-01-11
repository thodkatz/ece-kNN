#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "v0.h"
#include "utils.h"


/*
 * Calculate dot product using cblas routine
 *
 * 0 --> false
 * 1 --> true 
 */
#define CBLAS_DDOT 1

/*
 * Calculate transpose matrix using cblas routine
 *
 * 0 --> false
 * 1 --> true
 */
#define CBLAS_TRANS 1

/*
 * Create a transpose of distance matrix (nxm)
 *
 * Note: The data is stored in row major format.
 * For the knn, given the distance matrix (nxm) we need
 * to iterate each column for the k-select that will 
 * result to cache miss.
 *
 * Is it worth it to create first a transpose matrix?
 *
 * 0 --> don't create
 * 1 --> create
 *
 * Results: Much better performance to transpose first
 */
#define TRANS 0

/*
 * Use STL algorithms for k-select
 * 0 --> false
 * 1 --> true
 */
#define CPP 0

/*
 * Calculate euclidean distance matrix
 *
 *
 * 0 --> naive d^2 = (x_i - y_i)^2 Requires TRANS == 0
 * 1 --> D = sqrt(sum(X.^2,2) - 2 * X*Y.' + sum(Y.^2,2).') Requires TRANS == 1
 * 2 --> D' = sqrt(sum(X.^2,2).' - 2 * Y * X.' + sum(Y.^2, 2)) (mxn) Requires TRANS == 0
 */
#define MATRIX 2

#if CPP == 1
#include <algorithm> 
#include <unordered_map>
#endif

knnresult kNN(double *x, double *y, uint32_t n, uint32_t m, uint32_t d, uint32_t k) {

    /* printf("Entering knn function, n: %d, m:%d\n", n, m); */
    /* printf("The x is \n"); */
    /* print_dataset_yav(x, n, d); */
    /* printf("The y is \n"); */
    /* print_dataset_yav(y, m, d); */

    knnresult ret;
    MALLOC(double, ret.ndist, m*MIN(k, n));
    MALLOC(uint32_t, ret.nidx, m*MIN(k, n));
    ret.k = k; // included the query to be tested?
    ret.m = m;

    struct timespec tic;
    struct timespec toc;
    
    TIC()
#if MATRIX == 0
    double *distance = euclidean_distance_naive(x, y, n, d, m);
#elif MATRIX == 1
    double *distance = euclidean_distance(x, y, n, d, m);
#elif MATRIX == 2
    double *distance = euclidean_distance_notrans(x, y, n, d, m);
#endif
    TOC("Time elapsed calculating distance matrix (seconds): %lf\n")

#if MATRIX == 1
    /* printf("\nThe distance matrix n x m: \n"); */
    /* print_dataset_yav(distance, n, m); */
#elif MATRIX == 2
    /* printf("\nThe distance matrix m x n: \n"); */
    /* print_dataset_yav(distance, m, n); */
#endif


#if TRANS == 1
    double *distance_trans;
    if (n != m) {
        TIC()

        // create transpose to exploit cache hits 
        MALLOC(double, distance_trans, m*n);
#if CBLAS_TRANS == 0  
        transpose(distance, distance_trans, n, m);
#elif CBLAS_TRANS == 1 
        cblas_domatcopy(CblasRowMajor, CblasTrans, n, m, 1, distance, m, distance_trans, n); 
#endif

        free(distance);

        TOC("Time elapsed calculating transpose distance mxn matrix (seconds): %lf\n")

        /* printf("\nThe transpose distance matrix m x n: \n"); */
        /* print_dataset_yav(distance_trans, m, n); */
    }
    else distance_trans = distance; // symmetric
#endif

    int isFirst = 1; // indeces 1 or 0 based?
    printf("Sorting distance matrix until %uth element...\n", k);

    TIC()

    for(uint32_t i = 0; i < m; i++) {
#if TRANS == 1
        double *distance_per_row = distance_trans + i*n; 
#elif TRANS == 0
        double *distance_per_row; 
#if MATRIX == 1
        MALLOC(double, distance_per_row, n);
        for(uint32_t j = 0; j < n; j++) distance_per_row[j] = distance[i + j*m];
#elif MATRIX == 0 || MATRIX == 2
        distance_per_row = distance + i*n;
#endif
#endif
        uint32_t *indeces_per_row;
        MALLOC(uint32_t, indeces_per_row, n);
        for(uint32_t j = 0; j < n; j ++) indeces_per_row[j] = j + isFirst;

#if CPP == 1 
        if(n>k) {
            // map each distance value to the index of the corpus
            std::unordered_map<double, uint32_t> mth_map; 
            for (uint32_t j=0; j < n; j++) mth_map[distance_per_row[j]] = j; // costs time

            // find the k smallest
            std::nth_element(distance_per_row, distance_per_row + k-1, distance_per_row + n);
            //memcpy(ret.ndist + i*k, distance_per_row, sizeof(double)*k);

            for(uint32_t j = 0; j < k; j++) {
                ret.ndist[i*k + j] = distance_per_row[j]; 

                ret.nidx[i*k + j] = mth_map[distance_per_row[j]] + isFirst;

                // in case there are duplicate distance values
                auto range = mth_map.equal_range(distance_per_row[j]); 
                auto count = mth_map.count(distance_per_row[j]);
                for (auto it = range.first; it != range.second; it++) {
                    ret.nidx[i*k + j] = it->second + isFirst;
                    if(count-- != 1) {
                        if (j < k) j++; 
                        else break;
                    }
                }
            } 
        }
        else {
            memcpy(ret.ndist + i*n, distance_per_row, sizeof(double) * n);
            memcpy(ret.nidx + i*n, indeces_per_row, sizeof(uint32_t) * n);
        }

#elif CPP == 0 
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
#endif

#if TRANS == 0 && MATRIX == 1
        free(distance_per_row);
#endif
    }

#if TRANS == 1
    free(distance_trans);
#elif TRANS == 0
    free(distance);
#endif

    TOC("Time elapsed calculating kNN given distance matrix (seconds): %lf\n")

    /* if (k>n) k = n; */
    /* printf("\nDistance of kNN\n"); */
    /* print_dataset_yav(ret.ndist, m, k); */
    /* printf("\nIndeces of kNN\n"); */
    /* print_indeces(ret.nidx, m, k); */

    return ret;
}
